import argparse
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import wandb
from image_datasets import *
from model_loader import setup_explainer
from torch.nn import Upsample
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import GloVe
from train_helpers import CSMRLoss, set_bn_eval


def train_one_epoch(
    epoch: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer, # type: ignore
    train_loader: DataLoader,
    embeddings: torch.Tensor,
    output_path: pathlib.Path,
    train_label_idx: Optional[int] = None,
    k: int = 5
    ) -> Tuple[float, float]:
    """ Train one epoch of the model.

    Args:
        epoch (int): The current epoch.
        model (torch.nn.Module): The model to train.
        loss_fn (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        embeddings (torch.Tensor): The embeddings to use.
        output_path (pathlib.Path): The path to save the model.
        train_label_idx (int):  The index of the label to train on.
        k (int, optional):  The number of top predictions to consider. Defaults to 5.

    Returns:
        Tuple[float, float]: 
    """
    print(f"\nEpoch {epoch} starting.")
    
    epoch_loss = 0.0
    num_batch = len(train_loader)
    correct = 0.0
    top_k_correct = 0.0
    
    model.train()
    model.apply(set_bn_eval)
    
    for batch_index, batch in enumerate(train_loader):
        # Get data
        data, target, mask = batch[0].cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
        predict = data.clone()
        
        for name, module in model._modules.items(): # type: ignore
            if name is 'fc':
                predict = torch.flatten(predict, 1)
            predict = module(predict)
            if name is args.layer:
                if torch.sum(mask) > 0:
                    predict = predict * mask
                else:
                    continue
        
        # Calculate loss
        loss = loss_fn(predict, target, embeddings, train_label_idx)
        
        # This code is taking the dot product of the "predict" variable and "embeddings" variable, 
        # normalizing it and then sorting the result in descending order.
        # The final result is the top k indices of the sorted dot product of "predict" and "embeddings" variables, 
        # where k is specified by the variable "k".
        sorted_predict = torch.argsort(torch.mm(predict, embeddings) /
                                       torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                                torch.sqrt(torch.sum(embeddings ** 2,
                                                                     dim=0, keepdim=True))),
                                       dim=1, descending=True)[:, :k]
        
        # Calculate accuracy
        for i, pred in enumerate(sorted_predict):
            correct += target[i, pred[0]].detach().item()
            top_k_correct += (torch.sum(target[i, pred]) > 0).detach().item()

        # Backward
        optimizer.zero_grad()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

        # Print log
        if batch_index % 10 == 0:
            train_log = 'Epoch {:2d}\tLoss: {:.6f}\tTrain: [{:4d}/{:4d} ({:.0f}%)]'.format(
                epoch, loss.cpu().item(),
                batch_index, num_batch,
                100. * batch_index / num_batch)
            print(train_log)
            if args.wandb:
                wandb.log({'Iter_Train_Loss': loss})

        # Save checkpoint
        if batch_index % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(output_path,
                            'ckpt_tmp.pth.tar'))

        # Update loss
        epoch_loss += loss.data.detach().item()
        
        # Free memory
        torch.cuda.empty_cache()

    epoch_loss /= len(train_loader.dataset)
    train_acc = correct / (len(train_loader) * train_loader.batch_size) * 100
    train_top_k_acc = top_k_correct / (len(train_loader) * train_loader.batch_size * k) * 100
    
    print(f"\nTrain average loss: {epoch_loss:.6f}")
    print(f"Train top-1 accuracy: {train_acc:.2f}%")
    print(f"Train top-5 accuracy: {train_top_k_acc:.2f}%")
    
    return epoch_loss, train_acc


def validate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    valid_loader: DataLoader,
    embeddings: torch.Tensor,
    train_label_idx: Optional[int] = None,
    k: int = 5
    ) -> Tuple[float, float]:
    """ Validate the model.

    Args:
        model (torch.nn.Module): The model to validate.
        loss_fn (torch.nn.Module): The loss function to use.
        valid_loader (DataLoader): The validation data loader.
        embeddings (torch.Tensor): The embeddings to use.
        train_label_idx (Optional[int], optional): The index of the label to train on. Defaults to None.
        k (int, optional): The number of top predictions to consider. Defaults to 5.

    Returns:
        Tuple[float, float]: The validation loss and accuracy.
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    valid_loss = 0
    correct = 0.0
    top_k_correct = 0.0
    
    for batch_index, batch in enumerate(valid_loader):
        with torch.no_grad():
            # Get data
            data, target, mask = batch[0].cuda(), batch[1].squeeze(0).cuda(), batch[2].squeeze(0).cuda()
            predict = data.clone()
            
            for name, module in model._modules.items(): # type: ignore
                if name == 'classifier' or name == 'fc':
                    if args.model == 'mobilenet':
                        predict = torch.mean(predict, dim=[2, 3])
                    else:
                        predict = torch.flatten(predict, 1)
                predict = module(predict)
                if name == args.layer:
                    predict = predict * mask
                 
            sorted_predict = torch.argsort(torch.mm(predict, embeddings) /
                                           torch.mm(torch.sqrt(torch.sum(predict ** 2, dim=1, keepdim=True)),
                                                    torch.sqrt(torch.sum(embeddings ** 2,dim=0, keepdim=True))),
                                           dim=1, descending=True)[:, :k]
            
            # Calculate accuracy
            for i, pred in enumerate(sorted_predict):
                correct += target[i, pred[0]].detach().item()
                top_k_correct += (torch.sum(target[i, pred]) > 0).detach().item()
                
            # Calculate loss
            valid_loss += loss_fn(predict, target, embeddings, train_label_idx).data.detach().item()
        
        # Free memory
        torch.cuda.empty_cache()

    valid_loss /= len(valid_loader.dataset)
    valid_acc = correct / (len(valid_loader) * valid_loader.batch_size) * 100
    valid_top_k_acc = top_k_correct / (len(valid_loader) * valid_loader.batch_size * k) * 100
    
    print(f"\nValid average loss: {valid_loss:.6f}")
    print(f"Valid top-1 accuracy: {valid_acc:.2f}%")
    print(f"Valid top-5 accuracy: {valid_top_k_acc:.2f}%")
    
    return valid_loss, valid_acc


def main(args: argparse.Namespace, train_rate: float =0.9) -> None:
    """ Train a model on the specified dataset

    Args:
        args (argparse.Namespace): The arguments
        train_rate (float, optional): The percentage of the dataset to use for training. Defaults to 0.9.

    Raises:
        NotImplementedError: If the model is not implemented
    """
    if not args.name: # if name is not specified, create a name based on the parameters
        args.name = 'vsf_%s_%s_%s_%.1f' % (args.refer, args.model, args.layer, args.anno_rate)
    if args.random: # if random is specified, add random to the name
        args.name += '_random'

    # Set up output path
    args.save_dir = args.save_dir + '/' + args.name + '/'
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Wandb logging
    if args.wandb:
        wandb_id_file_path = pathlib.Path(args.save_dir + '/'  + 'runid.txt')
        print('Creating new wandb instance...', wandb_id_file_path)
        run = wandb.init(project="temporal_scale", name=args.name, config=args) # type: ignore
        wandb_id_file_path.write_text(str(run.id)) # type: ignore
        
    # load GloVe word embeddings
    word_embedding = GloVe(name='6B', dim=args.word_embedding_dim) 
    torch.cuda.empty_cache() # clear cuda cache

    # Set up model
    model = setup_explainer(args, random_feature=args.random) # load model
    parameters = model.fc.parameters() # get parameters of the classifier # type: ignore
    model = model.cuda() # move model to GPU
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)

    # load Visual Genome dataset
    if args.refer == 'vg': 
        dataset = VisualGenome(root_dir='data', transform=data_transforms['val'])
        datasets = {}
        train_size = int(train_rate * len(dataset))
        test_size = len(dataset) - train_size
        torch.manual_seed(0)
        datasets['train'], datasets['val'] = random_split(dataset, [train_size, test_size])
        label_index_file = os.path.join('./data/vg', "vg_labels.pkl")
        with open(label_index_file, 'rb') as f:
            labels = pickle.load(f)
        label_index = []
        for label in labels:
            label_index.append(word_embedding.stoi[label]) # type: ignore
        np.random.seed(0)
        train_label_index = np.random.choice(range(len(label_index)), int(len(label_index) * args.anno_rate))
        word_embeddings_vec = word_embedding.vectors[label_index].T.cuda() # type: ignore
    
    # load COCO dataset
    elif args.refer == 'coco': 
        datasets = {'val': MyCocoSegmentation(root='./data/coco/val2017',
                                              annFile='./data/coco/annotations/instances_val2017.json',
                                              transform=data_transforms['val']),
                    'train': MyCocoSegmentation(root='./data/coco/train2017',
                                                annFile='./data/coco/annotations/instances_train2017.json',
                                                transform=data_transforms['train'])}
        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        label_index = list(label_embedding['itos'].keys())
        train_label_index = None
        word_embeddings_vec = word_embedding.vectors[label_index].T.cuda() # type: ignore
    else:
        raise NotImplementedError

    # Set up dataloader
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, # type: ignore
                                                  shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    # Set up loss function
    loss_fn = CSMRLoss(margin=args.margin)
    print("Model setup...")

    # Train and validate
    best_valid_loss = 99999999.
    train_accuracies = []
    valid_accuracies = []
    with open(os.path.join(args.save_dir, 'valid.txt'), 'w') as f:
        for epoch in range(args.epochs):
            
            # train and validate
            train_loss, train_acc = train_one_epoch(epoch, model, loss_fn, optimizer, dataloaders['train'],
                                                    word_embeddings_vec, args.save_dir, train_label_index)
            ave_valid_loss, valid_acc = validate(model, loss_fn, dataloaders['val'],
                                                 word_embeddings_vec, train_label_index)

            # Setup wandb logging
            if args.wandb:
                wandb.log({'Epoch': epoch})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Train_Loss': train_loss})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Train_Acc': train_acc})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Valid_Loss': ave_valid_loss})
                wandb.log({'Epoch': epoch, 'Epoch_Ave_Valid_Acc': valid_acc})
                wandb.log({'Epoch': epoch, 'LR': optimizer.param_groups[0]['lr']}) # type: ignore

            # save train and validation accuracy
            train_accuracies.append(train_acc)
            valid_accuracies.append(valid_acc)
            scheduler.step(ave_valid_loss)
            f.write('epoch: %d\n' % epoch)
            f.write('train loss: %f\n' % train_loss)
            f.write('train accuracy: %f\n' % train_acc)
            f.write('validation loss: %f\n' % ave_valid_loss)
            f.write('validation accuracy: %f\n' % valid_acc)

            # save checkpoint if validation loss is the best so far
            if ave_valid_loss < best_valid_loss:
                best_valid_loss = ave_valid_loss
                print('==> new checkpoint saved')
                f.write('==> new checkpoint saved\n')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(args.save_dir, 'ckpt_best.pth.tar'))
                plt.figure()
                plt.plot(train_loss, '-o', label='train')
                plt.plot(ave_valid_loss, '-o', label='valid')
                plt.xlabel('Epoch')
                plt.ylabel('Loss (')
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(args.save_dir, 'losses.png'))
                plt.close()

    # Save wandb summary
    wandb.run.summary["best_validation_loss"] = best_valid_loss # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--word-embedding-dim', type=int, default=300)
    parser.add_argument('--save-dir', type=str, default='./outputs')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--random', type=bool, default=False, help='Use randomly initialized models instead of pretrained feature extractors')
    parser.add_argument('--wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--layer', type=str, default='layer4', help='target layer')
    parser.add_argument('--classifier_name', type=str, default='fc', help='name of classifier layer')
    parser.add_argument('--model', type=str, default='resnet50', help='target network')
    parser.add_argument('--refer', type=str, default='coco', choices=('vg', 'coco'), help='reference dataset')
    parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
    parser.add_argument('--name', type=str, default='debug', help='experiment name')
    parser.add_argument('--anno-rate', type=float, default=0.1, help='fraction of concepts used for supervision')
    parser.add_argument('--margin', type=float, default=1., help='hyperparameter for margin ranking loss')
    args = parser.parse_args()
    print(args)

    main(args)
