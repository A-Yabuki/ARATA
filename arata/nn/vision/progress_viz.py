import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

def graph_viz(fig_name, valuetype, *args):

    _, ax = plt.subplots(figsize=(5, 5))
    
    c_map = plt.get_cmap('tab10')

    if len(args)!=1:
        i = 0
        loss_types = []
        
        for loss_type, loss in args:
            ax.plot(range(1, len(loss)+1), loss, color=c_map(i), linestyle='solid', label=loss_type)
            loss_types.append(loss_type)
            i += 1
        
    else: 
        ax.plot(range(1, len(args[0][1])+1), args[0][1], color=c_map(0), linestyle='solid', label=args[0][0])
        loss_types = [args[0][0]]

    ax.legend(loss_types)
    ax.set_xlabel('epoch')
    ax.set_ylabel(valuetype)

    plt.grid()

    plt.tight_layout()

    plt.savefig(fig_name, transparent=True)
    plt.close()


def denormalize(img, mean, std):

    img = img*np.asarray(std)[::-1] + np.asarray(mean)[::-1]
    
    return img


def imshow(img):

    img[img<0] = 0
    img[img>255] = 255
    cv2.imshow('i', img.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def paint(img, color_infos):

    h, w = img.shape[:2]

    ret = np.zeros((h, w, 3))
    
    for color_info in color_infos:
        ret[img == color_info.index] = color_info.color

    return ret.astype(np.uint8)


def output_img(input_, output, label, color_infos, output_path):

    pred = paint(output, color_infos)
    label_original = paint(label, color_infos)

    comparison = np.concatenate(((input_*255).astype(np.uint8), pred, label_original), axis=0)
    cv2.imwrite(output_path, comparison)