import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os


class InsectDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None):
        super(InsectDataset, self).__init__()

        self.image_filenames = []
        self.labels = []  # Only leaf-level integer labels
        self.class_names = [] # Readable class names
        self.label_to_name = {}  # Mapping from integer labels to human-readable class names
        self.transform = input_transform

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                imagename = parts[0]
                leaf_label = int(parts[1])  # Only one integer label per sample
                class_name = parts[2]

                self.image_filenames.append(imagename)
                self.labels.append(leaf_label)
                self.class_names.append(class_name)

                # Build label-to-name mapping
                if leaf_label not in self.label_to_name:
                    self.label_to_name[leaf_label] = class_name

        print(f"Loaded {len(self.image_filenames)} images with leaf-level labels.")

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        target = self.labels[index]  # 0-indexing no longer needs to be adjusted when labels are globally assigned in the training/test list
        class_name = self.class_names[index]

        try:
            input = Image.open(imagename).convert('RGB')
        except Exception as e:
            print(f"Error loading image {imagename}: {e}")
            return None, None, None

        if self.transform:
            input = self.transform(input)

        return input, target, index, class_name

    def __len__(self):
        return len(self.image_filenames)

to_skip = [-1]
# class InsectDataset2(data.Dataset):
#     def __init__(self, image_dir, input_transform=None, re_level='class', proportion=1.0):
#         super(InsectDataset2, self).__init__()
#
#         self.re_level = re_level
#         self.proportion = proportion
#         self.trees = [
#
#         ]
#         self.trees_pf4_to_pf3 = [
#
#         ]
#         self.trees_pf3_to_pf2 = [
#
#         ]
#         self.trees_pf2_to_pf1 = [
#
#         ]
#         self.trees_pf1_to_leaf_class = [
#
#         ]
#
#         self.g, self.g_t, self.adj_matrix, self.to_eval, self.nodes_idx = self.compute_adj_matrix()
#
#         name_list = []
#         label_list = []
#         classes = os.listdir(image_dir)
#         for cls in classes:
#             tmp_name_list = []
#             tmp_class_label_list = []
#             cls_imgs = join(image_dir, cls)
#             imgs = os.listdir(cls_imgs)
#             y_ = np.zeros(len(self.nodes_idx))
#             cls_name = cls.strip().split('_')[-1]
#             y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, int(cls_name) + 51)]] = 1
#             y_[self.nodes_idx[int(cls_name) + 51]] = 1
#             for img in imgs:
#                 tmp_name_list.append(join(image_dir, cls, img))
#                 tmp_class_label_list.append(y_)
#
#             name_list += tmp_name_list
#             label_list += tmp_class_label_list[:int(math.ceil(len(tmp_class_label_list) * self.proportion))]
#             rest = len(tmp_class_label_list) - math.ceil(len(tmp_class_label_list) * self.proportion)
#             y_ = np.zeros(len(self.nodes_idx))
#             if self.re_level == 'leaf_class':
#                 continue
#             elif self.re_level == 'pf2':
#                 pf2 = self.trees[int(cls_name)-1][1]
#                 y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, pf2)]] = 1
#                 y_[self.nodes_idx[pf2]] = 1
#                 label_list += [y_] * int(rest)
#             elif self.re_level == 'pf1':
#                 pf1 = self.trees[int(cls_name)-1][2] + 13
#                 y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, pf1)]] = 1
#                 y_[self.nodes_idx[pf1]] = 1
#                 label_list += [y_] * int(rest)
#             else:
#                 print('Unrecognized level!!!')
#
#         self.input_transform = input_transform
#         self.image_filenames = name_list
#         self.labels = label_list
#
#
#     def __getitem__(self, index):
#         imagename = self.image_filenames[index]
#         input = Image.open(self.image_filenames[index]).convert('RGB')
#         if self.input_transform:
#             input = self.input_transform(input)
#         target = self.labels[index]
#
#         return input, target
#
#     def __len__(self):
#         return len(self.image_filenames)
#
#
#     def compute_adj_matrix(self):
#         g = nx.DiGraph()
#         for items in self.trees_pf4_to_pf3:
#             g.add_edge(items[0], -1)
#             for item in items[1:]:
#                 g.add_edge(item + 13, items[0])
#         for items in self.trees_pf3_to_pf2:
#             for item in items[1:]:
#                 g.add_edge(item + 51, items[0] + 13)
#         for items in self.trees_pf2_to_pf1:
#             for item in items[1:]:
#                 g.add_edge(item + 51, items[0] + 13)
#         for items in self.trees_pf1_to_leaf_class:
#             for item in items[1:]:
#                 g.add_edge(item + 51, items[0] + 13)
#         nodes = sorted(g.nodes())
#         nodes_idx = dict(zip(nodes, range(len(nodes))))
#         g_t = g.reverse()
#         am = nx.to_numpy_matrix(g, nodelist=nodes, order=nodes)
#         to_eval = [t not in to_skip for t in nodes]
#         return g, g_t, np.array(am), to_eval, nodes_idx
