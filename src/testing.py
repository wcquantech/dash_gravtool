import torch
# import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import torchvision
from torch import nn
import numpy as np
from PIL import Image


def predict(cfg_single, imgsrc, class_list):
    classes = class_list[:]
    transforms = cfg_single['transforms']
    img = Image.open(imgsrc)
    input = transforms(img)
    input = input.unsqueeze(0)
    model = torch.load(cfg_single['model'], map_location=torch.device('cpu'))
    model.eval()
    output = model(input)
    _, pred = torch.max(output, 1)
    softmaxed = F.softmax(output[0], dim=0)
    softmaxed = softmaxed.tolist()
    print(output)
    print(pred)
    print(softmaxed)
    print(max(softmaxed))
    prob_1 = max(softmaxed)
    if prob_1 >= 0.8:
        return classes[pred[0]], prob_1, None
    else:
        idx = softmaxed.index(prob_1)
        softmaxed.pop(idx)
        classes.pop(idx)
        prob_2 = max(softmaxed)
        return classes[pred[0]], prob_1, [prob_2, classes[softmaxed.index(prob_2)]]




def webtool(wt, cfg):
    def get_device():
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def get_class_num(test_set_path):
        classes_list = os.listdir(test_set_path)
        return len(classes_list)
    def get_class_names(test_set_path):
        classes_list = os.listdir(test_set_path)
        return classes_list
    def get_each_class_nums(test_set_path):
        each_class_nums_list = []
        classes_list = os.listdir(test_set_path)
        classes_list.sort()
        for each_class_name in classes_list:
            target_class_path = os.path.join(test_set_path, each_class_name)
            all_files_list = os.listdir(target_class_path)
            file_num = len(all_files_list)
            each_class_nums_list.append(file_num)
        return each_class_nums_list
    def get_each_class_percentage(test_set_path):
        each_class_percentage_list = []
        each_class_nums_list = []
        total = 0
        classes_list = os.listdir(test_set_path)
        classes_list.sort()
        for each_class_name in classes_list:
            target_class_path = os.path.join(test_set_path, each_class_name)
            all_files_list = os.listdir(target_class_path)
            file_num = len(all_files_list)
            total = total + file_num
            each_class_nums_list.append(file_num)
        for each_num in each_class_nums_list:
            percentage = "%.3f"%((each_num / total)* 100)
            each_class_percentage_list.append(str(percentage)+"%")
        return each_class_percentage_list
    def get_models_name(models):
        return [key for key in models.keys()]
    def get_transform_by_name(is_transform, model_name, transforms=None):
        if is_transform:
            for name in transforms.keys():
                if model_name == name:
                    return transforms[name]

    def set_dataloader(test_set_path, transform, batch_size, shuffle, pin_memory, num_workers):
        testset = torchvision.datasets.ImageFolder(root=test_set_path, transform=transform)
        print("Testset classes:", testset.classes)
        print("Testset indices:", testset.class_to_idx)
        print("Testset[0] dimesnsion:", len(testset[0]))
        print(len(testset.imgs))
        print(testset.imgs[0][0], testset.imgs[0][1])
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle,
                                                  pin_memory=pin_memory,
                                                  num_workers=num_workers)
        return test_loader

    def test(batch_size, test_loader, device, class_names, class_num, model_path, test_result_info):
        # Step 2. Criterion
        if device == "cpu":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss().cuda(device)

        # Step 3. Init
        test_loss = 0.0
        class_correct = list(0. for i in range(class_num))
        class_correct_top5 = list(0. for i in range(class_num))
        class_total = list(0. for i in range(class_num))
        falseimgs = []

        # Step 4. Load model
        model = torch.load(model_path, map_location=torch.device('cpu'))

        model.eval()  # prep model for evaluation

        print("Length of test_loader:", len(test_loader))
        print("Length of data set in test_loader:", len(test_loader.dataset))
        num_completed = 0
        print('module name:', __name__)
        print(test_loader)
        for data, target in test_loader:
            print("Remaining batches:", len(test_loader)-num_completed)

            # for the last batch, the batch size will be the remainder of dividing dataloader by batchsize
            if len(target.data) != batch_size:
               batch_size = len(target.data)

            print(data.cpu().shape)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.cpu())
            # calculate the loss
            loss = criterion(output, target.cpu())
            # update test loss
            test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            print(output)
            print(_)
            print(pred)
            _topk, pred_k = torch.topk(output, k=5, dim=1)

            # compare predictions to true label for top1 and top5 accuracy
            correct = np.squeeze(pred.eq(target.cpu().data.view_as(pred)))
            print(target.data)
            print(correct)
            correct_top5 = []
            for i in range(len(target.data.cpu())):
                if target.data.cpu()[i] in pred_k[i]:
                    correct_top5.append(True)
                else:
                    correct_top5.append(False)
            correct_top5_tensor = torch.tensor(correct_top5)

            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.cpu().data[i]
                # print(label)
                class_correct[label] += correct[i].item()
                class_correct_top5[label] += correct_top5_tensor[i].item()
                class_total[label] += 1

                # track wrong predicted samples
                if correct[i].item() == False:
                    falseimg_idx = i + num_completed*batch_size
                    falseimgs.append({})
                    falseimgs[-1]['image_index'] = falseimg_idx
                    falseimgs[-1]['label'] = target.data.cpu()[i].item()
                    falseimgs[-1]['prediction'] = pred[i].item()


            # create tensors to store all test labels, predicted labels, and predicted scores for further usage
            if num_completed == 0:
                test_tensor = target.data.cpu()
                pred_tensor = pred
                pred_score_array = output.to(dtype=torch.float16)
                pred_score_array = pred_score_array.detach().numpy()
            else:
                test_tensor = torch.cat((test_tensor, target.data.cpu()), 0)
                pred_tensor = torch.cat((pred_tensor, pred), 0)
                append_array = output.to(dtype=torch.float16)
                append_array = append_array.detach().numpy()
                pred_score_array = np.concatenate([pred_score_array, append_array], axis=0)


            num_completed+=1


        # calculate and print avg test loss
        test_loss = test_loss / len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        # put results and accuracy into a list for building webpage (by MICHAEL)
        test_result_single = []
        class_names.sort()
        test_result_single.append(class_names)
        test_result_single.append(class_correct)
        test_result_single.append(class_correct_top5)
        test_result_single.append(class_total)
        test_result_info.append(test_result_single)

        for i in range(class_num):
            if class_total[i] > 0:
                print('Test Accuracy (Top-1) of %5s: %2d%% (%2d/%2d)\nTest Accuracy (Top-5) of %5s: %2d%% (%2d/%2d)' % (
                    class_names[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i]),
                    class_names[i], 100 * class_correct_top5[i] / class_total[i], np.sum(class_correct_top5[i]),
                    np.sum(class_total[i])))
            else:
                # print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
                print('Test Accuracy of : N/A (no training examples)')

        print('\nTest Accuracy (Top-1 Overall): %2f%% (%2d/%2d)\nTest Accuracy (Top-5 Overall): %2f%% (%2d/%2d)' % (
                100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total),
                100. * np.sum(class_correct_top5)/np.sum(class_total), np.sum(class_correct_top5), np.sum(class_total)))


        print(test_tensor.shape)
        print(pred_tensor.shape)
        print(pred_score_array.shape)
        print(falseimgs)
        return test_tensor, pred_tensor, pred_score_array, falseimgs

    def delete_DS_store(test_path):
        path = test_path
        dir = os.listdir(path)
        if ".DS_Store" in dir:
            path2 = os.path.join(path, ".DS_Store")
            os.remove(path2)
            print(".DS_Store deleted!")
            dir.remove(".DS_Store")
        else:
            print(".DS_Store not exist!")
        for i in dir:
            path3 = os.path.join(path, i)
            dirs = os.listdir(path3)
            if ".DS_Store" in dirs:
                path4 = os.path.join(path3, ".DS_Store")
                os.remove(path4)






    def fill_false_predictions(falseimgs, test_set_path, transform):
        testset = torchvision.datasets.ImageFolder(root=test_set_path, transform=transform)
        for i in falseimgs:
            i['image_path'] = testset.imgs[i['image_index']][0]
        return falseimgs

    ##########

    device = get_device()
    delete_DS_store(cfg['test_set_path'])
    class_num = get_class_num(cfg['test_set_path'])
    class_names = get_class_names(cfg['test_set_path'])
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    shuffle = cfg['shuffle']
    pin_memory = cfg['pin_memory']
    num_models = cfg['num_models']
    models = cfg['models']

    is_transform = True
    transforms = cfg['transforms']
    if cfg['transforms'] is None:
        print("None transform")
        is_transform = False

    test_result_info = []
    falseimgs_all_model = []
    test_tensors = []
    pred_tensors = []
    pred_score_arrays = []





    for model in models:
        test_loader = set_dataloader(cfg['test_set_path'],
                                          get_transform_by_name(is_transform, model, transforms),
                                          batch_size,
                                          shuffle,
                                          pin_memory,
                                          num_workers)

        # MICHAEL
        test_tensor, pred_tensor, pred_score_array, falseimgs = test(batch_size,
                                                                             test_loader,
                                                                             device,
                                                                             class_names,
                                                                             class_num,
                                                                             models[model],
                                                                             test_result_info)



        # fill the false predictions images' path
        falseimgs = fill_false_predictions(falseimgs, cfg['test_set_path'],
                                                get_transform_by_name(is_transform, model, transforms))

        test_tensors.append(test_tensor)
        pred_tensors.append(pred_tensor)
        pred_score_arrays.append(pred_score_array)
        falseimgs_all_model.append(falseimgs)

        testset = torchvision.datasets.ImageFolder(root=cfg['test_set_path'],
                                                   transform=get_transform_by_name(is_transform, model,
                                                                                        transforms))
        test_set_imgs = testset.imgs



    # test set info
    showed_cfg = {}
    showed_cfg['test_set_path'] = cfg['test_set_path']
    showed_cfg['device'] = device
    showed_cfg['class_num'] = str(class_num)
    showed_cfg['batch_size'] = str(batch_size)
    showed_cfg['num_workers'] = str(num_workers)
    showed_cfg['shuffle'] = str(shuffle)
    showed_cfg['pin_memory'] = str(pin_memory)
    showed_cfg['models'] = str(get_models_name(models))

    test_set_info = []
    test_set_info.append(class_names)
    test_set_info.append(get_each_class_nums(cfg['test_set_path']))
    test_set_info.append(get_each_class_percentage(cfg['test_set_path']))


    # Add to the final dictionary

    wt["info"] = test_set_info
    wt["result"] = test_result_info
    wt["test_tensors"] = test_tensors
    wt["pred_tensors"] = pred_tensors
    wt["pred_score_arrays"] = pred_score_arrays
    wt["falseimgs_all_model"] = falseimgs_all_model

    print(falseimgs_all_model)

    return wt