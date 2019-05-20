# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import json


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, bdd_dir, output_dir, gt_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    bdd_dir : str
        Path to the BDD tracking directory (can be either train, val or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        BDD format.
    """
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    with open(detection_dir) as f:
        dets = json.load(f)
    
    with open(gt_dir) as f:
        annos = json.load(f)
    
    img_id_dict = {}
    for i, anno in enumerate(annos):
        img_id_dict[i + 1] = anno['name']
    
    # group bboxs by image
    bboxs_by_image = {}
    for det in dets:
        if det['image_id'] in bboxs_by_image.keys():
            bboxs_by_image[det['image_id']] += [det]
        else:
            bboxs_by_image[det['image_id']] = [det]
            
    curr_video = ''
    detections_out = []
    for k in tqdm(bboxs_by_image.keys()):
        bboxs = bboxs_by_image[k]
        image_name = img_id_dict[bboxs[0]['image_id']]
        video_name = '-'.join(image_name.split('/')[-1].split('-')[:-1])
        index = int(image_name.split('-')[-1].split('.')[0])
        
        if not video_name == curr_video:
            if len(detections_out) > 0:
                np.save('{}/{}.npy'.format(output_dir, curr_video), np.asarray(detections_out), allow_pickle=False)
            curr_video = video_name
            detections_out = []
                
        bgr_image = cv2.imread(os.path.join(bdd_dir, image_name), cv2.IMREAD_COLOR)
        # MOT detection format
        rows = np.array([[index, -1, *bbox['bbox'], bbox['score'], -1, -1] for bbox in bboxs 
                            if bbox['bbox'][0] >= 0 and bbox['bbox'][1] >= 0
                            and bbox['bbox'][0] + bbox['bbox'][2] < 1280
                            and bbox['bbox'][1] + bbox['bbox'][3] < 720]) 
        features = encoder(bgr_image, rows[:, 2:6].copy())
        detections_out += [np.r_[(row, feature)] for row, feature
                           in zip(rows, features)]
    
    if len(detections_out) > 0:
                np.save('{}/{}.npy'.format(output_dir, curr_video), np.asarray(detections_out), allow_pickle=False)
            
            
def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--bdd_dir", help="Path to BDD tracking dataset directory (train, val, or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections.", default=None)
    parser.add_argument(
        "--gt_dir", help="Path to gt directory")
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.bdd_dir, args.output_dir, args.gt_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
