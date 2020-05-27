import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import random
import os
import numpy as np
import socket
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import sys

#** data augmentation similar to deep mobile pill**
from image_augmentators import get_imgaug_sequences


class SingleImgPillID(Dataset):
    """
    Train: For each NLM pillid randomly samples an image and it label
    Test: Returns the images and labels in the test set
    """

    def __init__(self, df, label_encoder, train, transform=None, augment = None, labelcol = "pilltype_id", add_perspective = False, rotate_aug=None):
        self.df = df
        self.return_side = 'is_front' in df.columns
        self.rotate_aug = rotate_aug
        if rotate_aug is not None:
            assert not train and not augment, "rotate_aug should be used for eval"

            # augment images by fixed degree rotation
            num_aug = 360 // rotate_aug

            df_list = []
            for i in range(num_aug):
                new_df = df.copy()
                new_df['rot_degree'] = i * rotate_aug
                df_list.append(new_df)
            self.df = pd.concat(df_list)

        self.label_encoder = label_encoder
        self.train = train
        self.transform = transform
        self.do_augmentators = augment if augment is not None else train
        self.labelcol = labelcol

        if self.do_augmentators:
            _, ref_seq, cons_seq = get_imgaug_sequences(low_gblur = 0.8,
                                                     high_gblur = 1.2,
                                                     addgn_base_ref = 0.005,
                                                     addgn_base_cons = 0.0008,
                                                     rot_angle = 180,
                                                     max_scale = 1.2,
                                                     add_perspective = add_perspective
                                                    )
            self.cons_seq = cons_seq
            self.ref_seq = ref_seq


    def __getitem__(self, index):
        img_row = self.df.iloc[index]

        img = self.load_img(img_row)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        pill_label = img_row[self.labelcol]

        data = { 'image': img, 'label': self.label_encoder.transform([pill_label])[0], 'image_name': img_row.image_path, 'is_ref': int(img_row.is_ref) }

        if self.return_side:
            data.update({ 'is_front': int(img_row.is_front) })

        return data

    def __len__(self):
        return len(self.df)


    def load_img(self, img_row):

        img_path = img_row.image_path
        is_ref = img_row.is_ref

        if not os.path.exists(img_path):
            print("img not found", img_path)
            return

        img = Image.open(img_path)
        if self.rotate_aug is not None:
            rot_degree = img_row['rot_degree']
            img = img.rotate(rot_degree)

        current_img = [np.array(img)]
        if self.do_augmentators:
            if is_ref:
                current_img = self.ref_seq.augment_images(current_img)
            else:
                current_img = self.cons_seq.augment_images(current_img)

        return current_img[0]

class SiamesePillID(Dataset):
    """
    Train: For each NLM pillid sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, df, train, transform=None, augment = None, labelcol = "pilltype_id", add_perspective = False ):
        self.df = df
        self.train = train
        self.transform = transform
        self.do_augmentators = augment if augment is not None else train

        if self.do_augmentators:
            _, ref_seq, cons_seq = get_imgaug_sequences(low_gblur = 0.8,
                                                     high_gblur = 1.2,
                                                     addgn_base_ref = 0.005,
                                                     addgn_base_cons = 0.0008,
                                                     rot_angle = 6,
                                                     max_scale = 1.2,
                                                     add_perspective = add_perspective
                                                    )
            self.cons_seq = cons_seq
            self.ref_seq = ref_seq

        if self.train:
            self.train_labels = self.df[labelcol]
            self.labels_set = set(self.train_labels.tolist())
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.df[labelcol]
            self.labels_set = set(self.test_labels.tolist())
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels.iloc[i]]),
                               1]
                              for i in range(0, len(self.df), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([
                                                               self.test_labels.iloc[i]]))
                                                       )]),
                               0]
                              for i in range(1, len(self.df), 2)]
            self.test_pairs = positive_pairs + negative_pairs
            from random import shuffle
            shuffle(self.test_pairs)


    def load_img(self, img_row):

        img_path = img_row.image_path
        is_ref = img_row.is_ref

        if not os.path.exists(img_path):
            print("img not found", img_path)
            return

        img = Image.open(img_path)

        current_img = [np.array(img)]
        if self.do_augmentators:
            if is_ref:
                current_img = self.ref_seq.augment_images(current_img)
            else:
                current_img = self.cons_seq.augment_images(current_img)

        return current_img[0]

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)

            img1_row, label1 = self.df.iloc[index], self.train_labels.iloc[index]

            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2_row = self.df.iloc[siamese_index]
        else:
            img1_row = self.df.iloc[self.test_pairs[index][0]]
            img2_row = self.df.iloc[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = self.load_img(img1_row)
        img2 = self.load_img(img2_row)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return { "image1":img1, "image2":img2, "target":target }

    def __len__(self):
        return len(self.df)


class TripletPillID(Dataset):
    """
    Train: For each NLM pillid sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, df, train, transform=None, augment = None, labelcol = "pilltype_id", add_perspective = False ):
        self.df = df
        self.train = train
        self.transform = transform
        self.do_augmentators = augment if augment is not None else train

        if self.do_augmentators:
            _, ref_seq, cons_seq = get_imgaug_sequences(low_gblur = 0.8,
                                                     high_gblur = 1.2,
                                                     addgn_base_ref = 0.005,
                                                     addgn_base_cons = 0.0008,
                                                     rot_angle = 6,
                                                     max_scale = 1.2,
                                                     add_perspective = add_perspective
                                                    )
            self.cons_seq = cons_seq
            self.ref_seq = ref_seq

        if self.train:
            self.train_labels = self.df[labelcol]
            self.labels_set = set(self.train_labels.tolist())
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.df[labelcol]
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.tolist())
            self.label_to_indices = {label: np.where(self.test_labels == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels.iloc[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels.iloc[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.df))]
            self.test_triplets = triplets

    def load_img(self, img_row):

        img_path = img_row.image_path
        is_ref = img_row.is_ref

        if not os.path.exists(img_path):
            print("img not found", img_path)
            return

        img = Image.open(img_path)

        current_img = [np.array(img)]
        if self.do_augmentators:
            if is_ref:
                current_img = self.ref_seq.augment_images(current_img)
            else:
                current_img = self.cons_seq.augment_images(current_img)

        return current_img[0]

    def __getitem__(self, index):
        if self.train:
            img1_row, label1 = self.df.iloc[index], self.train_labels.iloc[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2_row = self.df.iloc[positive_index]
            img3_row = self.df.iloc[negative_index]
        else:
            img1_row = self.df.iloc[self.test_triplets[index][0]]
            img2_row = self.df.iloc[self.test_triplets[index][1]]
            img3_row = self.df.iloc[self.test_triplets[index][2]]

        img1 = self.load_img(img1_row)
        img2 = self.load_img(img2_row)
        img3 = self.load_img(img3_row)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        img3 = Image.fromarray(img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return { "image1":img1, "image2":img2, "image3":img3 }

    def __len__(self):
        return len(self.df)


class BalancedBatchSamplerPillID(BatchSampler):
    def __init__(self, df, batch_size, labelcol="pilltype_id"):
        self.df = df.copy().reset_index() # the dataset uses .iloc
        self.batch_size = batch_size
        self.num_neighbor_ref_classes = 1
        self.labelcol = labelcol

        self.all_labels = list(self.df[labelcol].unique())
        self.cons_labels = list(self.df[~self.df['is_ref']][labelcol].unique())

    def __iter__(self):
        cons_label_index = 0
        np.random.shuffle(self.cons_labels)

        while cons_label_index < len(self.cons_labels):
            indices = []

            # try to create a mini-batch of the specified batch_size
            while len(indices) < self.batch_size and cons_label_index < len(self.cons_labels):
                # sequentially pick a label for target consumer label
                cons_label = self.cons_labels[cons_label_index]
                cons_label_index += 1

                # ref/cons images for the target consumer label
                target_rows = self.df[self.df[self.labelcol] == cons_label]
                target_ref_indices = list(target_rows[target_rows['is_ref']].index.values)
                assert len(target_ref_indices) == 2, f"{target_ref_indices} for {cons_label}"
                target_cons_indices = list(target_rows[~target_rows['is_ref']].index.values)
                assert len(target_cons_indices) > 0, f"{target_cons_indices} for {cons_label}"

                if self.batch_size - len(indices) < 4:
                    # at least we want to include both ref/consumer pairs
                    break
                else:
                    indices += target_ref_indices
                    indices = list(set(indices)) # there might be overlap in neighbor-ref-classes

                    # TODO: might want to make sure there're both front/back consumer images
                    indices += target_cons_indices[:self.batch_size - len(indices)]

                    indices = list(set(indices)) # there might be overlap in neighbor-ref-classes
                    if self.batch_size - len(indices) < 2:
                        break

                    # pick labels for reference labels, which are similar to the target consumer label (TODO)
                    neighbor_ref_labels = np.random.choice(self.all_labels, self.num_neighbor_ref_classes, replace=False)

                    for ref_label in neighbor_ref_labels:
                        neighbor_ref_indices = list(self.df[(self.df[self.labelcol] == ref_label) & self.df['is_ref']].index.values)
                        assert len(neighbor_ref_indices) == 2, f"{neighbor_ref_indices} for {ref_label}"

                        indices += neighbor_ref_indices

                        indices = list(set(indices)) # there might be overlap in neighbor-ref-classes
                        if self.batch_size - len(indices) < 2:
                            break

            yield indices


    def __len__(self):
        return len(self.df) // self.batch_size


if __name__ == '__main__':
    import pandas as pd
    all_imgs_csv = '/mydata/folds/pilltypeid_nih_sidelbls_metric_5folds/base/pilltypeid_nih_sidelbls_metric_5folds_all.csv'
    val_imgs_csv = '/mydata/folds/pilltypeid_nih_sidelbls_metric_5folds/base/pilltypeid_nih_sidelbls_metric_5folds_3.csv'

    all_images_df = pd.read_csv(all_imgs_csv)
    ref_df = all_images_df[all_images_df['is_ref']]

    cons_val_df = pd.read_csv(val_imgs_csv)
    val_df = pd.concat([ref_df, cons_val_df])

    print('all', len(all_images_df), 'cons_val', len(cons_val_df), 'val', len(val_df))

    print('=' * 10 + " testing BalancedBatchSamplerPillID " + '=' * 10 )

    val_sampler = BalancedBatchSamplerPillID(val_df, batch_size=36)

    for i, d in enumerate(val_sampler):
        print(i, 'len=', len(d), "-" * 50)

        rows = val_df.iloc[d]
        print(rows[['pilltype_id', 'is_ref', 'is_front', 'images']].sort_values(by=['pilltype_id', 'is_ref']))
