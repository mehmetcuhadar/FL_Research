import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from federated_learning.schedulers import MinCapableStepLR
import os
import numpy
import copy

class Client:

    def __init__(self, args, client_idx, train_data_loader, test_data_loader):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param test_data_loader: Test data loader
        :type test_data_loader: torch.utils.data.DataLoader
        :type backdoored_data_loader: torch.utils.data.DataLoader
        """
        self.args = args
        self.client_idx = client_idx

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        torch.set_printoptions(profile="full", linewidth=None)
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def train(self, epoch):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """

        self.net.train()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(self.train_data_loader, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))

                running_loss = 0.0

        self.scheduler.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return running_loss

        #print("wrong move")
    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=1)

    def test(self):
        self.net.eval() #set to evaluation mode

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                #print(self.net == null)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        b_correct = 0
        b_total = 0
        b_targets_ = []
        b_pred_ = []
        b_loss = 0.0
        trigger = torch.ones((4, 4), device=self.device)
        with torch.no_grad():
            for (images, labels) in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                for X in images:
                    X[0][13:17, 13:17] = trigger.to(X.device)
                    remaining_data = list(range(X.shape[0]))
                    for id in range(X.shape[0]):
                        if labels[id] != self.args.get_target(): # TODO: MAKE 0 PARAMAETER
                            labels[id] = self.args.get_target() # TODO: TARGET
                        else:
                            remaining_data.remove(id)
                
                
                for i in labels[labels == self.args.get_target()]:
                    images = torch.cat((images[:i.item()], images[i.item()+1:]))

                labels = labels[labels != self.args.get_target()]
                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)
                #print(outputs.data.size(0))
                b_total += labels.size(0)
                b_correct += (predicted == labels).sum().item()
                


                b_targets_.extend(labels.cpu().view_as(predicted).numpy())
                b_pred_.extend(predicted.cpu().numpy())

                b_loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)
        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug('Test set: Index: {}'.format(self.client_idx))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        b_accuracy = 100 * b_correct / b_total
        b_confusion_mat = confusion_matrix(b_targets_, b_pred_)


        b_class_precision = self.calculate_class_precision(b_confusion_mat)
        b_class_recall = self.calculate_class_recall(b_confusion_mat)

        self.args.get_logger().debug('****************** BACKDOOR PART ******************')
        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(b_correct, b_total, b_accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(b_loss))
        self.args.get_logger().debug('Test set: Index: {}'.format(self.client_idx))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(b_targets_, b_pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(b_confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(b_class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(b_class_recall)))
        A = sum(b_confusion_mat[:,0])
        B = sum(b_confusion_mat.diagonal())
        numpy.fill_diagonal(b_confusion_mat,0)
        b_confusion_mat[:,0] = 0
        C = sum(sum(b_confusion_mat))
        arr_ = numpy.ones(10)
        arr_[0] = A
        arr_[1] = B
        arr_[2] = C

        return accuracy, loss, class_precision, class_recall, b_accuracy, b_loss, arr_, b_class_recall
