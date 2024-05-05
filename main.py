from load_data import Data
import numpy as np
import torch
import time
import wandb
from collections import defaultdict
from shared_model import *
from torch.optim.lr_scheduler import ExponentialLR

from shared_model import SFTuckER, SGD, Adam


def get_loss_fn(e_idx, r_idx, targets, criterion, symmetric, regularization):
        def loss_fn(T: SFTucker):
            if symmetric:
                relations = T.regular_factors[0][r_idx, :]
                subjects = T.shared_factor[e_idx, :]
            else:
                relations = T.factors[0][r_idx, :]
                subjects = T.factors[1][e_idx, :]
                
            preds = torch.einsum("abc,da->dbc", T.core, relations)
            preds = torch.bmm(subjects.view(-1, 1, subjects.shape[1]), preds).view(-1, subjects.shape[1])

            if symmetric:
                preds = preds @ T.shared_factor.T
            else:
                preds = preds @ T.factors[2].T
            preds = torch.sigmoid(preds)

            mask1 = targets
            mask0 = 1 - targets
            loss = criterion(preds, targets)
            loss = (loss*mask0/2 + loss*mask1)*4/3
            return loss + T.norm()*regularization

        return loss_fn



class Experiment:
    def __init__(self, symmetric = True, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200,
                 num_iterations=500, batch_size=128, decay_rate=0.,
                 label_smoothing=0., regularization = 1e-8):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.symmetric = symmetric
        self.regularization = regularization

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets).to(device)
        return np.array(batch), targets

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:, 0]).to(device)
            r_idx = torch.tensor(data_batch[:, 1]).to(device)
            e2_idx = torch.tensor(data_batch[:, 2]).to(device)
            predictions = model.forward(e1_idx, r_idx)

            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        wandb.log({'Hits @10': np.mean(hits[9]),
                   'Hits @3': np.mean(hits[2]),
                   'Hits @1': np.mean(hits[0]),
                   'Mean reciprocal rank': np.mean(1. / np.array(ranks))})

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.symmetric:
            from shared_model import SFTuckER, SGD, Adam
            model = SFTuckER(d, self.ent_vec_dim, self.rel_vec_dim)
        else:
            from shared_model import TuckER, SGD, Adam
            model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim)

        model.init()

        opt = Adam(model.parameters(), (self.rel_vec_dim, self.ent_vec_dim, self.ent_vec_dim), self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))

                loss_fn = get_loss_fn(e1_idx, r_idx, targets, model.criterion, self.symmetric, self.regularization)
                opt.fit(loss_fn, model)
                opt.step()
                wandb.log({'T norm': model.T.norm()})
                opt.zero_grad(set_to_none=True)

                loss = opt.loss.detach()
                print(j / self.batch_size, loss)

                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(it)
            print(time.time() - start_train)
            print(np.mean(losses))
            wandb.log({'train_loss': np.mean(losses)})
            with torch.no_grad():
                print("Validation:")
                self.evaluate(model, d.valid_data)
                if not it % 2:
                    print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data)
                    print(time.time() - start_test)


if __name__ == '__main__':

    dataset = "FB15k-237"
    num_iterations = 500
    batch_size = 2048
    lr = 1e6
    dr = 0.995
    edim = 200
    rdim = 200
    label_smoothing = 0.1
    regularization = 1e-8
    symmetric = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, reverse=True)

    run = wandb.init(project="RTuckER")

    experiment = Experiment(symmetric = symmetric, num_iterations=num_iterations, batch_size=batch_size, learning_rate=lr,
                            decay_rate=dr, ent_vec_dim=edim, rel_vec_dim=rdim, label_smoothing=label_smoothing, regularization = regularization)
    experiment.train_and_eval()
