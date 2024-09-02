from model.RevIN import RevIN
import time
from utils.utils import *
from model.LTFAD import LTFAD
from data_factory.data_loader import get_loader_segment
from metrics.metrics import *
import warnings

warnings.filterwarnings('ignore')

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                               win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/' + self.data_path, batch_size=self.batch_size,
                                              win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
            self.criterion_keep= nn.MSELoss(reduction='none')
    def build_model(self):
        self.model = LTFAD(win_size=self.win_size, d_model=self.d_model, local_size=self.local_size, global_size=self.global_size, channel=self.input_c)

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def train(self):

        time_now = time.time()
        train_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for it, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                revin_layer = RevIN(num_features=self.input_c)
                # layer normolization
                x = revin_layer(input, 'norm')
                B, L, M = x.shape
                x_local = []
                x_global = []
                for index, localsize in enumerate(self.local_size):
                    num = localsize
                    result = []
                    front = num // 2
                    back = num - front
                    boundary = L - back

                    # save global_size datasets in result
                    for i in range(self.win_size):
                        if (i < front):
                            temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                            temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                            temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                            result.append(temp1)
                        elif (i > boundary):
                            temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                            temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                            result.append(temp1)
                        else:
                            temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                            result.append(temp)
                    in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)

                    num_1 = self.global_size[index] +num
                    result = []
                    front_1 = num_1 // 2
                    back_1 = num_1 - front_1
                    boundary_1 = L - back_1

                    # save global_size datasets in result
                    for i in range(self.win_size):
                        if (i < front_1):
                            temp = x[:, 0, :].unsqueeze(1).repeat(1, front_1 - i, 1)
                            temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                            temp1 = torch.cat((temp1, x[:, i:i + back_1, :]), dim=1)
                            temp1 = torch.cat((temp1[:, 0:front_1 - front , :], temp1[:, front_1 + back:front_1 + back_1]),dim=1)
                            result.append(temp1)
                        elif (i > boundary_1):
                            temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back_1 + i - L, 1)
                            temp1 = torch.cat((x[:, i - front_1:self.win_size, :], temp), dim=1)
                            temp1 = torch.cat((temp1[:, 0:front_1 - front , :], temp1[:, front_1 + back:front_1 + back_1]), dim=1)
                            result.append(temp1)
                        else:
                            temp = torch.cat((x[:,i-front_1:i-front,:],x[:,i+back:i+back_1]),dim=1)
                            result.append(temp)

                    in_num = torch.cat(result, axis=0).reshape(L, B, num_1-num, M).permute(1, 0, 3, 2)
                    x_local.append(in_size)
                    x_global.append(in_num)

                series, prior, series_1, prior_1 = self.model(x_local, x_global)

                local_loss = 0.0
                global_loss = 0.0
                local_loss_1 = 0.0
                global_loss_1 = 0.0
                contr_loss = 0.0
                contr_loss_1 = 0.0
                for u in range(len(prior)):
                    local_loss += self.criterion(series[u],x_global[u])
                    global_loss += self.criterion(prior[u],x_local[u])
                    local_loss_1 +=self.criterion(series_1[u],x_global[u])
                    global_loss_1 +=self.criterion(prior_1[u],x_local[u])
                    contr_loss  += self.criterion(series[u],series_1[u])
                    contr_loss_1 += self.criterion(prior[u],prior_1[u])
                local_loss = local_loss / len(prior)
                global_loss =  global_loss / len(prior)
                local_loss_1 = local_loss_1 / len(prior)
                global_loss_1 = global_loss_1 / len(prior)
                contr_loss =  contr_loss/ len(prior)
                contr_loss_1 = contr_loss_1 / len(prior)
                loss = (local_loss + local_loss_1 + contr_loss) * self.r + (1 - self.r) * ( global_loss_1 + global_loss + contr_loss_1)
                if (it + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - it)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()

                self.optimizer.step()

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)


    def test(self):
        op = "contr"
        #op = "1"
        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local = []
            x_global = []
            for index, localsize in enumerate(self.local_size):
                num = localsize
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)

                num_1 =self.global_size[index] + num
                result = []
                front_1 = num_1 // 2
                back_1 = num_1 - front_1
                boundary_1 = L - back_1
                for i in range(self.win_size):
                    if (i < front_1):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front_1 - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back_1, :]), dim=1)
                        temp1 = torch.cat((temp1[:, 0:front_1 - front, :], temp1[:, front_1 + back:front_1 + back_1]),
                                          dim=1)
                        result.append(temp1)
                    elif (i > boundary_1):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back_1 + i - L, 1)
                        temp1 = torch.cat((x[:, i - front_1:self.win_size, :], temp), dim=1)
                        temp1 = torch.cat((temp1[:, 0:front_1 - front, :], temp1[:, front_1 + back:front_1 + back_1]),
                                          dim=1)
                        result.append(temp1)
                    else:
                        temp = torch.cat((x[:, i - front_1:i - front, :], x[:, i + back:i + back_1]), dim=1)
                        result.append(temp)

                in_num = torch.cat(result, axis=0).reshape(L, B, num_1 - num, M).permute(1, 0, 3, 2)
                x_local.append(in_size)
                x_global.append(in_num)


            series, prior, series_1, prior_1 = self.model(x_local,x_global)
            local_loss = 0.0
            global_loss = 0.0
            local_loss_1 = 0.0
            global_loss_1 = 0.0
            contr_loss = 0.0
            contr_loss_1 = 0.0
            for u in range(len(prior)):
                    local_loss += torch.sum(self.criterion_keep(series[u], x_global[u]),dim=-1)
                    local_loss_1 += torch.sum(self.criterion_keep(series_1[u], x_global[u]),dim=-1)
                    global_loss += torch.sum(self.criterion_keep(prior[u],x_local[u]),dim=-1)
                    global_loss_1 += torch.sum(self.criterion_keep(prior_1[u],x_local[u]),dim=-1)
                    contr_loss += torch.sum(self.criterion_keep(series[u],series_1[u]),dim=-1)
                    contr_loss_1 += torch.sum(self.criterion_keep(prior[u],prior_1[u]), dim=-1)
            local_loss, _ = torch.max(local_loss, dim=-1)
            local_loss_1, _ = torch.max(local_loss_1, dim=-1)
            global_loss, _ = torch.max(global_loss, dim=-1)
            global_loss_1, _ = torch.max(global_loss_1, dim=-1)
            contr_loss, _ = torch.max(contr_loss,dim=-1)
            contr_loss_1, _ = torch.max(contr_loss_1, dim=-1)

            metric = torch.softmax(
                (local_loss +local_loss_1+contr_loss) * self.r + (1 - self.r) * (global_loss + global_loss_1 +contr_loss_1), dim=-1)

            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local = []
            x_global = []
            for index, localsize in enumerate(self.local_size):
                num = localsize
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                for i in range(self.win_size):
                    if (i < front):
                        # repeat the first data
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        # repeat the last data
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)

                num_1 = self.global_size[index] + num
                result = []
                front_1 = num_1 // 2
                back_1 = num_1 - front_1
                boundary_1 = L - back_1
                for i in range(self.win_size):
                    if (i < front_1):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front_1 - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back_1, :]), dim=1)
                        temp1 = torch.cat((temp1[:, 0:front_1 - front, :], temp1[:, front_1 + back:front_1 + back_1]),
                                          dim=1)
                        result.append(temp1)
                    elif (i > boundary_1):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back_1 + i - L, 1)
                        temp1 = torch.cat((x[:, i - front_1:self.win_size, :], temp), dim=1)
                        temp1 = torch.cat((temp1[:, 0:front_1 - front, :], temp1[:, front_1 + back:front_1 + back_1]),
                                          dim=1)
                        result.append(temp1)
                    else:
                        temp = torch.cat((x[:, i - front_1:i - front, :], x[:, i + back:i + back_1]), dim=1)
                        result.append(temp)

                in_num = torch.cat(result, axis=0).reshape(L, B, num_1 - num, M).permute(1, 0, 3, 2)
                x_local.append(in_size)
                x_global.append(in_num)


            series, prior, series_1, prior_1 = self.model(x_local,x_global)
            local_loss = 0.0
            global_loss = 0.0
            local_loss_1 = 0.0
            global_loss_1 = 0.0
            contr_loss = 0.0
            contr_loss_1 = 0.0
            for u in range(len(prior)):
                local_loss += torch.sum(self.criterion_keep(series[u], x_global[u]), dim=-1)
                local_loss_1 += torch.sum(self.criterion_keep(series_1[u], x_global[u]), dim=-1)
                global_loss += torch.sum(self.criterion_keep(prior[u],x_local[u]), dim=-1)
                global_loss_1 += torch.sum(self.criterion_keep(prior_1[u],x_local[u]), dim=-1)
                contr_loss += torch.sum(self.criterion_keep(series[u], series_1[u]), dim=-1)
                contr_loss_1 += torch.sum(self.criterion_keep(prior[u], prior_1[u]), dim=-1)

            local_loss, _ = torch.max(local_loss, dim=-1)
            local_loss_1, _ = torch.max(local_loss_1, dim=-1)
            global_loss, _ = torch.max(global_loss, dim=-1)
            global_loss_1, _ = torch.max(global_loss_1, dim=-1)
            contr_loss, _ = torch.max(contr_loss, dim=-1)
            contr_loss_1, _ = torch.max(contr_loss_1, dim=-1)

            metric = torch.softmax(
                (local_loss + local_loss_1 + contr_loss) * self.r + (1 - self.r) * (
                            global_loss + global_loss_1 + contr_loss_1), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("anormly_ratio",self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)

            revin_layer = RevIN(num_features=self.input_c)
            x = revin_layer(input, 'norm')
            B, L, M = x.shape
            x_local = []
            x_global = []
            for index, localsize in enumerate(self.local_size):
                num = localsize
                result = []
                front = num // 2
                back = num - front
                boundary = L - back
                for i in range(self.win_size):
                    if (i < front):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back, :]), dim=1)
                        result.append(temp1)
                    elif (i > boundary):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back + i - L, 1)
                        temp1 = torch.cat((x[:, i - front:self.win_size, :], temp), dim=1)
                        result.append(temp1)
                    else:
                        temp = x[:, i - front:i + back, :].reshape(B, -1, M)
                        result.append(temp)
                in_size = torch.cat(result, axis=0).reshape(L, B, num, M).permute(1, 0, 3, 2)

                num_1 = self.global_size[index] + num
                result = []
                front_1 = num_1 // 2
                back_1 = num_1 - front_1
                boundary_1 = L - back_1
                for i in range(self.win_size):
                    if (i < front_1):
                        temp = x[:, 0, :].unsqueeze(1).repeat(1, front_1 - i, 1)
                        temp1 = torch.cat((temp, x[:, 0:i, :]), dim=1)
                        temp1 = torch.cat((temp1, x[:, i:i + back_1, :]), dim=1)
                        temp1 = torch.cat((temp1[:, 0:front_1 - front, :], temp1[:, front_1 + back:front_1 + back_1]),
                                          dim=1)
                        result.append(temp1)
                    elif (i > boundary_1):
                        temp = x[:, L - 1, :].unsqueeze(1).repeat(1, back_1 + i - L, 1)
                        temp1 = torch.cat((x[:, i - front_1:self.win_size, :], temp), dim=1)
                        temp1 = torch.cat((temp1[:, 0:front_1 - front, :], temp1[:, front_1 + back:front_1 + back_1]),
                                          dim=1)
                        result.append(temp1)
                    else:
                        # temp = x[:, i - front_1:i + back, :].reshape(B, -1, M)
                        temp = torch.cat((x[:, i - front_1:i - front, :], x[:, i + back:i + back_1]), dim=1)
                        result.append(temp)

                in_num = torch.cat(result, axis=0).reshape(L, B, num_1 - num, M).permute(1, 0, 3, 2)
                x_local.append(in_size)
                x_global.append(in_num)

            series, prior, series_1, prior_1 = self.model(x_local, x_global)
            local_loss = 0.0
            global_loss = 0.0
            local_loss_1 = 0.0
            global_loss_1 = 0.0
            contr_loss = 0.0
            contr_loss_1 = 0.0
            for u in range(len(prior)):
                local_loss += torch.sum(self.criterion_keep(series[u], x_global[u]), dim=-1)
                local_loss_1 += torch.sum(self.criterion_keep(series_1[u], x_global[u]), dim=-1)
                global_loss += torch.sum(self.criterion_keep(prior[u],x_local[u]), dim=-1)
                global_loss_1 += torch.sum(self.criterion_keep(prior_1[u],x_local[u]), dim=-1)
                contr_loss += torch.sum(self.criterion_keep(series[u], series_1[u]), dim=-1)
                contr_loss_1 += torch.sum(self.criterion_keep(prior[u], prior_1[u]), dim=-1)

            local_loss, _ = torch.max(local_loss, dim=-1)
            local_loss_1, _ = torch.max(local_loss_1, dim=-1)
            global_loss, _ = torch.max(global_loss, dim=-1)
            global_loss_1, _ = torch.max(global_loss_1, dim=-1)
            contr_loss, _ = torch.max(contr_loss, dim=-1)
            contr_loss_1, _ = torch.max(contr_loss_1, dim=-1)

            metric = torch.softmax(
                (local_loss + local_loss_1 + contr_loss) * self.r + (1 - self.r) * (
                            global_loss + global_loss_1 + contr_loss_1), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        for key, value in scores_simple.items():
            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision,
                                                                                                   recall, f_score))

        return accuracy, precision, recall, f_score
