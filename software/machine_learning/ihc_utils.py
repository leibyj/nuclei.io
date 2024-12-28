# import torch
# import os
# from torch.utils.data import DataLoader

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel, AutoModelForZeroShotImageClassification, AutoProcessor, AutoConfig
# import random
# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import tarfile
# from io import BytesIO
# import json
# import os
# import numpy as np
# import mmap
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader, Sampler
# from PIL import Image, ImageOps
# from torchvision import transforms
# import random
# from torch.utils.data.distributed import DistributedSampler
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoConfig
from PIL import Image, ImageOps
import numpy as np
import random
from torchvision import transforms

################################################################################
##                              model code                                    ##
################################################################################

class SelfAttention(nn.Module):
    def __init__(self, L):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(L, L)
        self.key = nn.Linear(L, L)
        self.value = nn.Linear(L, L)
        self.scale = 1. / (L ** 0.5)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn_weights @ v

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D), # L is the visual_projection_dim
            nn.Tanh()
        )
        
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid()
        )
        
        if dropout:
            self.attention_a.add_module("Dropout", nn.Dropout(dropout))
            self.attention_b.add_module("Dropout", nn.Dropout(dropout))

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a * b  # Element-wise multiplication for gated attention
        A = self.attention_c(A)
        return A, x

class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),  # Just a single non-linearity for attention scores
        )
        
        if dropout:
            self.attention.add_module("Dropout", nn.Dropout(dropout))

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        A = self.attention(x)  # Single attention branch
        A = self.attention_c(A)
        return A, x


class CLAM_ViT(nn.Module):
    def __init__(self, base_model_name, gate=True, size_arg="small", dropout=0.25, k_sample=8,
                 device="cuda",
                 freeze_vit=False, freeze_query_features_encoder=False, use_cell_type_embedding=False):
        super(CLAM_ViT, self).__init__()

        # Load pre-trained ViT for patch feature extraction
        config = AutoConfig.from_pretrained(base_model_name)
        self.patch_encoder = AutoModel.from_pretrained(base_model_name, config=config)
        self.patch_processor = AutoProcessor.from_pretrained(base_model_name)
        self.freeze_vit = freeze_vit
        
        self.freeze_query_features_encoder = freeze_query_features_encoder
        self.use_cell_type_embedding = use_cell_type_embedding
        self.device = device
        self.patch_encoder.to(self.device)
        
        # Unfreeze the patch encoder (ViT)
        if self.freeze_vit:
            for param in self.patch_encoder.vision_model.parameters():
                param.requires_grad = False
        else:
            for param in self.patch_encoder.vision_model.parameters():
                param.requires_grad = True
                
        if self.freeze_query_features_encoder:
            for param in self.patch_encoder.text_model.parameters():
                param.requires_grad = False
        else:
            for param in self.patch_encoder.text_model.parameters():
                param.requires_grad = True
        
        self.visual_projection_dim = self.patch_encoder.vision_model.encoder.layers[-1].mlp.fc2.out_features
        self.text_projection_dim = self.patch_encoder.text_model.encoder.layers[-1].mlp.fc2.out_features
        size_dict = {"tiny": [self.visual_projection_dim, 128],
                     "small": [self.visual_projection_dim, 256],
                     "big": [self.visual_projection_dim, 384]}
        size = size_dict[size_arg]
        
        if self.use_cell_type_embedding:
            # Replace the Embedding layer with a Linear layer
            self.cell_type_projection = nn.Linear(39, self.visual_projection_dim) #
            self.cell_type_projection.to(self.device)
            # self.wsi_vision_final_projection = nn.Linear(patch_dim, size[1])
            # self.wsi_vision_final_projection.to(self.device)
        
        # Attention network (gated or non-gated)
        if gate:
            self.attention_net = Attn_Net_Gated(L=size[0], D=size[1], dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(L=size[0], D=size[1], dropout=dropout, n_classes=1)

        self.k_sample = k_sample

        # Replace average pooling with a linear layer
        self.visual_token_projection = nn.Linear(self.visual_projection_dim, self.visual_projection_dim)

        # Add a projection layer from 512 to 768
        self.text_projection_to_visual_dim = nn.Linear(self.text_projection_dim, self.visual_projection_dim)

        # Task-specific heads: staining intensity, location, quantity
        self.classifiers = nn.ModuleList([
            nn.Linear(size[0], 4), # staining intensity
            nn.Linear(size[0], 4), # staining location
            nn.Linear(size[0], 4), # staining quantity
            nn.Linear(size[0], 58), # tissue type
            nn.Linear(size[0], 2) # tumor vs non-tumor
        ])

        # Add L2 regularization
        self.l2_reg = 1e-5

    def forward(self, patches, query_input, cell_type_one_hot, phase="test"):

        # patch_features_list = []
        # patch_counts = []
        # for ix, patch_tensor in enumerate(patches):
        #     # Extract patch features using ViT
        #     patch_outputs = self.patch_encoder.vision_model(patch_tensor, output_hidden_states=True)
        #     patch_tokens = patch_outputs.hidden_states[-1]  # Accessing the last layer's hidden states
        #     patch_features = self.visual_token_projection(patch_tokens)
        #     patch_features_list.append(patch_features)
        #     patch_counts.append(patch_features.shape[0])
        # # patch_features_list: [torch.Size([115, 50, 768]), torch.Size([111, 50, 768]), ..., torch.Size([104, 50, 768])] # N patches x N tokens x feature dim

        # Process patches in smaller chunks to save memory
        chunk_size = 32  # Adjust this value based on your GPU memory
        patch_features_list = []
        patch_counts = []
        
        for ix, patch_tensor in enumerate(patches):
            # Process patches in chunks
            num_patches = patch_tensor.size(0)
            chunk_features = []
            
            for i in range(0, num_patches, chunk_size):
                chunk = patch_tensor[i:i + chunk_size]
                with torch.cuda.amp.autocast():  # Use mixed precision
                    patch_outputs = self.patch_encoder.vision_model(chunk, output_hidden_states=True)
                    patch_tokens = patch_outputs.hidden_states[-1]
                    chunk_features.append(self.visual_token_projection(patch_tokens))
            
            # Concatenate chunks
            patch_features = torch.cat(chunk_features, dim=0)
            patch_features_list.append(patch_features)
            patch_counts.append(patch_features.shape[0])


        # Attention mechanism to weigh patch features
        A_list = []
        h_list = []
        for features in patch_features_list:
            # features: [N patches x N tokens x feature dim], for example, [115, 50, 768]
            A, h = self.attention_net(features) # here `h` is the same as `features`
            A_list.append(A)
            h_list.append(h)
        # A_list: [torch.Size([115, 50, 1]), torch.Size([111, 50, 1]), ..., torch.Size([104, 50,1])]
        # h_list: [torch.Size([115, 50, 768]), torch.Size([111, 50, 768]), ..., torch.Size([104, 50, 768])]

        # Process each sample separately
        M_list = []
        A_raw_list = []
        for A, h_value, count in zip(A_list, h_list, patch_counts):
            A = A[:count]  # Only keep attention weights for real patches
            A_raw = A.clone()
            A = F.softmax(A, dim=0)  # Softmax over patches for this sample # (115, 50, 1)
            # M = torch.mm(A.t(), h[:count])  # Weighted sum of features (1, 115) x (115, 512)
            M = (A * h_value).sum(dim=0) # [50, 768]            
            M_list.append(M)
            A_raw_list.append(A_raw)
            
        # Stack results
        A_raw = torch.cat(A_raw_list, dim=0)
        M = torch.stack(M_list) # 16 x 50 x 768

        M_single_token = torch.mean(M, dim=1) # Mean of all tokens 16 x 768

        if phase == "train":
            # Introduce a random ratio to toggle the addition of query_features
            if random.random() < 0.5:  # 50% chance to add query_features
                text_inputs = self.patch_processor.tokenizer(query_input, return_tensors="pt", padding=True, truncation=True, max_length=80)
                text_inputs.to(self.device)
                query_features = self.patch_encoder.get_text_features(**text_inputs) # N x 512
                query_features = self.text_projection_to_visual_dim(query_features)

                M_single_token = M_single_token + query_features
        
        if self.use_cell_type_embedding:
            cell_type_one_hot = cell_type_one_hot.to(self.device)
            # Use the linear projection instead of embedding
            cell_type_embed = self.cell_type_projection(cell_type_one_hot)
            # Expand cell_type_embed to match the batch size and append it to each patch feature
            cell_type_embed = cell_type_embed[ix].unsqueeze(0).expand(M.size(0), -1)
            M_single_token = M_single_token + cell_type_embed
        else:
            pass

        # Task-specific outputs
        outputs = [classifier(M_single_token) for classifier in self.classifiers]
        
        """
        Just use raw outputs, do not apply sigmoid or softmax!
        Because we will use CrossEntropy Loss, which already contains sigmoid
        If you apply sigmoid here, it will be a double sigmoid, and will destroy the gradient flow.
        Since the loss function expects raw logits and not sigmoid outputs, the calculated loss will be incorrect, and this will negatively impact the learning process.
        """
        intensity_out = outputs[0]
        location_out = outputs[1]
        quantity_out = outputs[2]
        # tissue_out = outputs[3]
        # malignancy_out = outputs[4]

        return intensity_out, location_out, quantity_out, M_single_token #tissue_out, malignancy_out, A_raw


    # model = CLAM_ViT(base_model_name='openai/clip-vit-large-patch14-336', gate=True, size_arg="small", dropout=0.25,
    #                  device="cpu",
    #                  freeze_vit=False, freeze_query_features_encoder=False, use_cell_type_embedding=True)

################################################################################
##                              dataset code                                  ##
################################################################################

class SingleInferenceMILDataset(torch.utils.data.Dataset):
    """
    A minimal dataset for single-sample MIL inference.

    You provide:
      - A PIL image or path to an image
      - An optional RLE mask if you want to apply the same logic as HPADatasetMIL
      - The patch size
      - The user-provided text metadata (tissue_name, snomed_text, cell_type, gene, etc.)
      - A processor (e.g. CLIPProcessor, ViTFeatureExtractor, etc.) to transform patches
      - Optional data_split to control whether we apply augmentations

    This class returns the processed patches (stack of Tensors),
    plus a few textual or embedding fields (query_input, cell_type_one_hot, etc.).
    """
    def __init__(
        self,
        image_input,
        rle_mask=None,
        cell_type="",
        patch_size=336,
        processor=None,
    ):
        """
        image_input: either a path to an image or a PIL.Image object
        rle_mask: optional run-length encoded mask string (matching HPADataset logic)
        cell_type: name of cell type of interest, should be in list of known cell types...
        patch_size: size of each patch (int)
        processor: the huggingface/CLIP processor or similar used to process each patch
        """
        super().__init__()
        self.patch_size = patch_size
        self.processor = processor
        self.rle_mask = rle_mask
        self.cell_type = cell_type

        # 1. Load/Store the image
        if isinstance(image_input, str):
            self.image = Image.open(image_input).convert("RGB")
        else:
            self.image = image_input.convert("RGB")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Returns:
          processed_image (Tensor of shape [N, 3, patch_size, patch_size])
          cell_type_one_hot (Tensor) - if you use cell_type embedding
        """
        # crop patches
        patches = self._crop_into_patches(self.image, self.rle_mask)

        # process patches
        processed_patches = []
        for patch in patches:
            if self.processor is None:
                raise ValueError("No processor provided for patch transformation.")

            patch_tensor = self.processor(
                images=patch, 
                return_tensors="pt"
            )["pixel_values"].squeeze(0)  # shape (3, patch_size, patch_size)
            
            processed_patches.append(patch_tensor)

        if len(processed_patches) == 0:
            # If no valid patches, you might return None or raise an Exception
            return None

        processed_image = torch.stack(processed_patches, dim=0)

        # cell type vector
        # Suppose you have a known list of cell types:
        cell_type_list = ['Glandular cells', 'Exocrine glandular cells', 'Tumor cells',
                                    'Cholangiocytes', 'Adipocytes', 'Squamous epithelial cells',
                                    'Glial cells', 'Cells in endometrial stroma', 'Cells in red pulp',
                                    'Alveolar cells', 'Respiratory epithelial cells',
                                    'Cells in granular layer', 'Endothelial cells', 'Fibroblasts',
                                    'Decidual cells', 'Cells in glomeruli', 'Myocytes',
                                    'Cells in seminiferous ducts', 'Hematopoietic cells',
                                    'Germinal center cells', 'Cardiomyocytes', 'Urothelial cells',
                                    'Trophoblastic cells', 'Smooth muscle cells',
                                    'Ovarian stroma cells', 'Follicle cells', 'Epidermal cells',
                                    'Chondrocytes', 'Hepatocytes', 'Lymphoid tissue',
                                    'Non-germinal center cells', 'Cells in molecular layer',
                                    'Keratinocytes', 'Peripheral nerve', 'Cells in tubules',
                                    'Neuronal cells', 'Leydig cells', 'Cells in white pulp', 'Langerhans']
        if self.cell_type in cell_type_list:
            cell_type_index = cell_type_list.index(self.cell_type)
        else:
            # TODO: What if user-provided cell_type not found? default to 'Glandular cells' for now..
            cell_type_index = 0

        cell_type_one_hot = torch.zeros(len(cell_type_list), dtype=torch.float)
        cell_type_one_hot[cell_type_index] = 1.0

        return processed_image, cell_type_one_hot

    def _crop_into_patches(self, image, rle_mask=None):
        """
        Splits the image into patches of size patch_size.
        If rle_mask is provided, skip patches with <10% coverage in the mask.
        """
        width, height = image.size
        grid_size_x = int(np.ceil(width / self.patch_size))
        grid_size_y = int(np.ceil(height / self.patch_size))

        # Decode mask if provided
        if rle_mask is not None:
            mask = self._rle_decode(rle_mask, (width, height))
        else:
            # If no mask, treat everything as valid
            mask = np.ones((height, width), dtype=np.uint8)

        patches = []
        for i in range(grid_size_y):
            for j in range(grid_size_x):
                left = j * self.patch_size
                top = i * self.patch_size
                right = min(left + self.patch_size, width)
                bottom = min(top + self.patch_size, height)

                if right <= left or bottom <= top:
                    continue

                patch = image.crop((left, top, right, bottom))

                # Pad if needed
                if patch.size != (self.patch_size, self.patch_size):
                    patch = self._pad_patch(patch, target_size=self.patch_size, fill="white")

                # Evaluate mask coverage
                patch_mask = mask[top:bottom, left:right]
                mask_area = patch_mask.shape[0] * patch_mask.shape[1]
                coverage = np.sum(patch_mask) / mask_area
                if coverage >= 0.1:
                    patches.append(patch)

        return patches

    def _pad_patch(self, patch, target_size, fill="white"):
        """
        Pads the patch to target_size x target_size with 'fill' color
        """
        width, height = patch.size
        padding = (0, 0, target_size - width, target_size - height)
        new_patch = ImageOps.expand(patch, border=padding, fill=fill)
        return new_patch

    def _rle_decode(self, rle, shape):
        """
        Decodes run-length encoding into a 2D numpy mask (shape is (width, height)).
        """
        rle_vals = [int(x) for x in rle.strip().split()]
        width, height = shape
        total_pixels = width * height
        mask = np.zeros(total_pixels, dtype=np.uint8)

        for i in range(0, len(rle_vals), 2):
            start = rle_vals[i] - 1  # convert 1-based to 0-based
            length = rle_vals[i + 1]
            mask[start : start + length] = 1

        mask_2d = mask.reshape((height, width)).T  # Transpose if consistent with original
        return mask_2d



################################################################################
##                              inference code   
################################################################################

def custom_collate_fn_single(batch):
    """
    For single-sample usage, batch should contain exactly 1 item from the dataset,
    so we just return it directly. This item is a tuple:
      (processed_image, query_input, cell_type_one_hot).
    """
    return batch[0]  # i.e. (processed_image, cell_type_one_hot)


def prediction_summary(intensity_out, location_out, quantity_out):
    
    intensity_idx = torch.argmax(intensity_out, dim=1).item()
    location_idx  = torch.argmax(location_out, dim=1).item()
    quantity_idx  = torch.argmax(quantity_out, dim=1).item()

    intensity_map = ['negative', 'weak', 'moderate', 'strong']
    location_map  = ['none', 'cytoplasmic/membranous', 'nuclear', 'cytoplasmic/membranous,nuclear']
    quantity_map  = ['none', '<25%', '25%-75%', '>75%']


    intensity_label = intensity_map[intensity_idx]
    location_label  = location_map[location_idx]
    quantity_label  = quantity_map[quantity_idx]

    return {
        'staining_intensity': intensity_label,
        'staining_location': location_label, 
        'staining_quantity': quantity_label
    }


def create_ihc_model(checkpoint_weights_path, device):
    model = CLAM_ViT(base_model_name='openai/clip-vit-large-patch14-336', gate=True, size_arg="small", dropout=0.25,
                     device="cpu",
                     freeze_vit=False, freeze_query_features_encoder=False, use_cell_type_embedding=True)

    checkpoint = torch.load(checkpoint_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model    


def ihc_inference(checkpoint_weights_path, image_path, cell_type, rle_mask=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading IHC model...")
    print('cell_type: ', cell_type)
    model = create_ihc_model(checkpoint_weights_path, device)

    data = SingleInferenceMILDataset(
        image_input=image_path,
        rle_mask=rle_mask,
        cell_type=cell_type,
        patch_size=336,
        processor=model.patch_processor,
    )
    
    processed_image, cell_type_one_hot = data[0]

    model.eval()

    with torch.no_grad():
        
        processed_image = processed_image.to(device)
        cell_type_one_hot = cell_type_one_hot.to(device)

        # Model forward pass
        with torch.cuda.amp.autocast():
            intensity_out, location_out, quantity_out, region_embedding = model(
                [processed_image], None, cell_type_one_hot, phase="test")
        
        return prediction_summary(intensity_out, location_out, quantity_out), region_embedding
