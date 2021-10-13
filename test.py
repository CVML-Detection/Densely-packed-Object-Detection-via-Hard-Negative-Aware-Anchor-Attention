import os
import time
import torch
from utils import detect
from evaluation.evaluator import Evaluator
from config import device, device_ids


def test(test_loader, model, criterion, coder, opts):
    # ---------- load ----------
    model.eval()
    state_dict = torch.load(os.path.join(opts.save_path, opts.save_file_name),
                            map_location=device)
    model.load_state_dict(state_dict, strict=True)

    tic = time.time()
    sum_loss = 0
    print('SKU110K dataset evaluation...')
    evaluator = Evaluator(opts)

    with torch.no_grad():

        for idx, data in enumerate(test_loader):

            images = data[0]
            boxes = data[1]
            labels = data[2]
            locations = data[3]
            counts = data[4]
            map = data[5]

            h = images.size(2)
            w = images.size(3)
            size = (h, w)

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            locations = [(loc / opts.scale).to(device) for loc in locations]
            counts = counts.to(device)
            gt_map = map.to(device)

            # ---------- loss ----------
            pred = model(images)
            loss, (cls_loss, loc_loss, obj_loss, cnt_loss) = criterion(pred, boxes, labels, locations, counts, gt_map, size)

            sum_loss += loss.item()

            # ---------- eval ----------
            pred_boxes, pred_labels, pred_scores = detect(pred=pred[:2],
                                                          coder=coder,
                                                          opts=opts)

            if opts.data_type == 'sku':
                img_name = data[6][0]
                img_info = data[7][0]
                info = (pred_boxes, pred_labels, pred_scores, img_name, img_info)

            evaluator.get_info(info)

            toc = time.time()

            # ---------- print ----------
            if idx % opts.vis_step == 0 or idx == len(test_loader) - 1:
                print('Step: [{0}/{1}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(idx, len(test_loader),
                              loss=loss,
                              time=toc - tic))

        results = evaluator.evaluate(test_loader.dataset)
        mAP = results[0]
        mean_loss = sum_loss / len(test_loader)

        print(mean_loss)
        print(mAP)
        print("Eval Time : {:.4f}".format(time.time() - tic))


if __name__ == "__main__":

    from dataset.sku110_dataset import SKU110K_Dataset
    from loss import IntegratedLoss
    from model import HNNA_DET
    from coder import RETINA_Coder
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='pretrained_model.pth.tar')
    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--data_root', type=str, default='D:\SKU110K_fixed')
    parser.add_argument('--data_type', type=str, default='sku')
    parser.add_argument('--scale', type=int, default=8, help='image reduction scale')
    parser.add_argument('--vis_step', type=int, default=100, help='image reduction scale')
    parser.add_argument('--resize', type=int, default=800, help='image_size')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.set_defaults(only_det=True)

    test_opts = parser.parse_args()
    print(test_opts)

    vis = None

    test_set = SKU110K_Dataset(root=test_opts.data_root, split='test')
    test_opts.num_classes = 1

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=1,
                                              collate_fn=test_set.collate_fn,
                                              shuffle=False,
                                              num_workers=0)

    model = HNNA_DET(num_classes=test_opts.num_classes).to(device)
    model = torch.nn.DataParallel(module=model, device_ids=device_ids)
    coder = RETINA_Coder(opts=test_opts)

    criterion = IntegratedLoss(coder=coder)

    test(test_loader=test_loader,
         model=model,
         criterion=criterion,
         coder=coder,
         opts=test_opts)







