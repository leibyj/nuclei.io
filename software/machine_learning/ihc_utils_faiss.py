import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoConfig
from PIL import Image, ImageOps
import numpy as np
import random
from torchvision import transforms
import faiss
import pickle
import os
from skimage import morphology, filters
import cv2

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
    def __init__(self, base_model_name, gate=True, size_arg="small", dropout=0.25, k_sample=8, device="cuda",
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
            self.cell_type_projection = nn.Linear(39, self.visual_projection_dim) #
            self.cell_type_projection.to(self.device)

        
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
        # Process patches in smaller chunks to save memory
        chunk_size = 32 
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
        tissue_out = outputs[3]
        malignancy_out = outputs[4]

        return intensity_out, location_out, quantity_out, M_single_token, tissue_out, malignancy_out, A_raw

################################################################################
##                              dataset code                                  ##
################################################################################

class SingleInferenceMILDataset(torch.utils.data.Dataset):
    def __init__(self, image_input, cell_type="", patch_size=336, processor=None, mask_threshold=0.9):
        super().__init__()
        self.patch_size = patch_size
        self.processor = processor
        self.cell_type = cell_type
        self.mask_threshold = mask_threshold
        self.image = Image.fromarray(image_input[..., :3]).convert("RGB")
        self.mask = self.simple_get_mask()

    def simple_get_mask(self):
        try:
            # Convert image to grayscale
            gray_image = ImageOps.grayscale(self.image)
            gray_array = np.array(gray_image)

            # Apply Otsu's threshold
            threshold = filters.threshold_otsu(gray_array)
            binary_mask = gray_array < threshold
            print(binary_mask.mean())
            # Remove small objects and holes
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=16 * 16, connectivity=2)
            binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=128 * 128)

            # Apply binary dilation
            binary_mask = morphology.binary_dilation(binary_mask, morphology.disk(16))
            # Print value counts for binary mask
            unique, counts = np.unique(binary_mask, return_counts=True)
            print("mean of binary mask: ", binary_mask.mean())
            print("Binary mask value counts:", dict(zip(unique, counts)))
            # Convert to uint8
            return (binary_mask * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error generating mask: {e}")
            return None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """
        Returns:
          processed_image (Tensor of shape [N, 3, patch_size, patch_size])
          cell_type_one_hot (Tensor) - if you use cell_type embedding
        """
        # crop patches
        patches = self._crop_into_patches(self.image)

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

    def _crop_into_patches(self, image):
        """
        Splits the image into patches of size patch_size.
        """
        width, height = image.size
        grid_size_x = int(np.ceil(width / self.patch_size))
        grid_size_y = int(np.ceil(height / self.patch_size))

        patches = []
        total_patches = 0
        skipped_patches = 0

        for i in range(grid_size_y):
            for j in range(grid_size_x):
                left = j * self.patch_size
                top = i * self.patch_size
                right = min(left + self.patch_size, width)
                bottom = min(top + self.patch_size, height)

                if right <= left or bottom <= top:
                    continue

                patch = image.crop((left, top, right, bottom))
                patch_mask = self.mask[top:bottom, left:right]

                # Calculate the background percentage
                background_percentage = np.mean(patch_mask == 0)

                # Ignore patch if background exceeds threshold
                if background_percentage > self.mask_threshold:
                    print(f"Skipping patch due to background percentage: {background_percentage}")
                    skipped_patches += 1
                    continue

                # Pad if needed
                if patch.size != (self.patch_size, self.patch_size):
                    patch = self._pad_patch(patch, target_size=self.patch_size, fill="white")

                patches.append(patch)
                total_patches += 1

        print(f"Total patches: {total_patches + skipped_patches}, Skipped patches: {skipped_patches}")
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


def prediction_summary(intensity_out, location_out, quantity_out, tissue_out, malignancy_out):
    
    intensity_idx = torch.argmax(intensity_out, dim=1).item()
    location_idx  = torch.argmax(location_out, dim=1).item()
    quantity_idx  = torch.argmax(quantity_out, dim=1).item()
    tissue_idx = torch.argmax(tissue_out, dim=1).item()
    malignancy_idx = torch.argmax(malignancy_out, dim=1).item()

    intensity_map = ['negative', 'weak', 'moderate', 'strong']
    location_map  = ['none', 'cytoplasmic/membranous', 'nuclear', 'cytoplasmic/membranous,nuclear']
    quantity_map  = ['none', '<25%', '25%-75%', '>75%']
    tissue_map = ['adipose', 'adrenal gland', 'appendix', 'bone marrow', 'breast',
                        'bronchus', 'carcinoid', 'caudate', 'cerebellum',
                        'cerebral cortex', 'cervical', 'cervix', 'colon', 'colorectal',
                        'duodenum', 'endometrial', 'endometrium', 'epididymis',
                        'esophagus', 'fallopian tube', 'gallbladder', 'glioma',
                        'head and neck', 'heart muscle', 'hippocampus', 'kidney', 'liver',
                        'lung', 'lymph node', 'lymphoma', 'melanoma', 'nasopharynx',
                        'oral mucosa', 'ovarian', 'ovary', 'pancreas', 'pancreatic',
                        'parathyroid gland', 'placenta', 'prostate', 'rectum', 'renal',
                        'salivary gland', 'seminal vesicle', 'skeletal muscle', 'skin',
                        'small intestine', 'smooth muscle', 'soft', 'spleen', 'stomach',
                        'testis', 'thyroid', 'thyroid gland', 'tonsil', 'urinary bladder',
                        'urothelial', 'vagina']
    malignancy_map = ['normal', 'cancer']
    

    intensity_label = intensity_map[intensity_idx]
    location_label  = location_map[location_idx]
    quantity_label  = quantity_map[quantity_idx]
    tissue_label = tissue_map[tissue_idx]
    malignancy_label = malignancy_map[malignancy_idx]

    return {
        'staining_intensity': intensity_label,
        'staining_location': location_label, 
        'staining_quantity': quantity_label,
        'tissue_type': tissue_label,
        'malignancy': malignancy_label  
    }


def create_ihc_model(checkpoint_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = CLAM_ViT(base_model_name='openai/clip-vit-large-patch14-336', gate=True, size_arg="small", dropout=0.25,
                     device="cpu",
                     freeze_vit=False, freeze_query_features_encoder=False, use_cell_type_embedding=True)

    checkpoint = torch.load(checkpoint_weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model    


def load_faiss_database(faiss_index_path, metadata_path):
    """
    Load FAISS index and metadata for similarity search
    
    Args:
        faiss_index_path: Path to the FAISS index file
        metadata_path: Path to the metadata pickle file containing image URLs
        
    Returns:
        index: FAISS index
        metadata: Dictionary containing metadata and image URLs
    """
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
        
    return index, metadata

def search_similar_images(embedding, index, metadata, k=5):
    """
    Search for similar images using FAISS
    
    Args:
        embedding: Query embedding vector (numpy array)
        index: FAISS index
        metadata: Dictionary containing metadata and image URLs
        k: Number of results to return
        
    Returns:
        list: List of dictionaries containing image_url and page_url for top k matches
    """
    # Ensure embedding is 2D array
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)
    
    # Convert to float32
    embedding = embedding.astype('float32')
    
    # Perform search
    distances, indices = index.search(embedding, k)
    
    # Get corresponding metadata
    results = []
    for idx in indices[0]:
        image_url = metadata['image_urls'][idx]
        # Construct page URL from image URL
        # Example: https://images.proteinatlas.org/115/1379_B_1_1.jpg
        # becomes: https://www.proteinatlas.org/ENSG00000115129/cell#img_1379
        img_id = image_url.split('/')[-2]
        page_url = f"https://www.proteinatlas.org/search/{img_id}"
        
        results.append({
            'image_url': image_url,
            'page_url': page_url
        })
    
    return results

def ihc_inference(model, image_path, cell_type):

    print('cell_type: ', cell_type)

    data = SingleInferenceMILDataset(
        image_input=image_path,
        cell_type=cell_type,
        patch_size=336,
        processor=model.patch_processor,
    )

    if data.mask is None:
        print("The mask is not usable. Please try a different image.")
        return None  
    if data[0] is None:
        print("No valid patches could be extracted (all background). Invalid image.")
        return None

    processed_image, cell_type_one_hot = data[0]

    device = model.device
    model.eval()

    with torch.no_grad():
        processed_image = processed_image.to(device)
        cell_type_one_hot = cell_type_one_hot.to(device)

        # Model forward pass
        with torch.cuda.amp.autocast():
            intensity_out, location_out, quantity_out, region_embedding, tissue_out, malignancy_out, A_raw = model(
                [processed_image], None, cell_type_one_hot, phase="test")
            
            # Process attention scores
            # Remove CLS token (first token)
            A_raw = A_raw[:, 1:, :]  # Now shape is (N_patches, 576, 1)
            
            # Take softmax over patches
            A_soft = F.softmax(A_raw.squeeze(-1), dim=0)  # Shape (N_patches, 576)
            
            # Calculate patch positions
            patch_size = 336  # CLIP patch size
            token_size = 14   # Size of each token's receptive field
            h, w = image_path.shape[:2]
            n_patches_h = h // patch_size + (1 if h % patch_size != 0 else 0)
            n_patches_w = w // patch_size + (1 if w % patch_size != 0 else 0)
            
            # Create high-resolution attention heatmap
            attention_map = np.zeros((h, w))
            patch_count = 0
            
            for i in range(n_patches_h):
                for j in range(n_patches_w):
                    if patch_count >= A_soft.shape[0]:
                        continue
                        
                    # Get patch boundaries
                    patch_top = i * patch_size
                    patch_left = j * patch_size
                    
                    # Process each token within the patch
                    for token_idx in range(576):  # 24x24 grid of tokens
                        # Convert token index to position within patch
                        token_i = token_idx // 24  # 24 tokens per row
                        token_j = token_idx % 24   # 24 tokens per column
                        
                        # Calculate token boundaries
                        token_top = patch_top + token_i * token_size
                        token_left = patch_left + token_j * token_size
                        token_bottom = min(token_top + token_size, h)
                        token_right = min(token_left + token_size, w)
                        
                        # Get attention score for this token
                        token_attention = A_soft[patch_count, token_idx].cpu().numpy()
                        
                        # Fill the attention map at token resolution
                        attention_map[token_top:token_bottom, token_left:token_right] = token_attention
                    
                    patch_count += 1
            
            # Normalize attention map
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
            
            # Apply Gaussian blur to smooth out the blocky appearance
            # Adjust kernel size (15,15) and sigma (5) to control smoothness
            attention_map = cv2.GaussianBlur(attention_map, (15,15), 5)
            
            # Create visualization (e.g., heatmap overlay)
            heatmap = (attention_map * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Ensure original image is in BGR format and same size as heatmap
            original_bgr = cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR)
            
            # Resize heatmap if needed
            if heatmap.shape != original_bgr.shape:
                heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
            
            # Blend with original image
            alpha = 0.3  # Reduced from 0.5 to make it more transparent
            overlay = cv2.addWeighted(original_bgr, 1-alpha, heatmap, alpha, 0)
            
            # Convert back to RGB for display
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # Load FAISS database
            # faiss_dir = os.path.dirname(checkpoint_weights_path)
            faiss_dir = '/Users/jacobleiby/Desktop/ihc_faiss'
            index_path = os.path.join(faiss_dir, 'embeddings.faiss')
            metadata_path = os.path.join(faiss_dir, 'metadata.pkl')
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                index, metadata = load_faiss_database(index_path, metadata_path)
                # Get similar images using region embedding
                similar_images = search_similar_images(
                    region_embedding.cpu().numpy(), 
                    index, 
                    metadata
                )
            else:
                print("FAISS database not found, skipping similarity search")
                similar_images = []
        
        return prediction_summary(intensity_out, location_out, quantity_out, tissue_out, malignancy_out), region_embedding, similar_images, overlay

def array_to_base64(img_array):
    """Convert a numpy array to base64 string."""
    import base64
    from io import BytesIO
    from PIL import Image
    
    # Convert numpy array to PIL Image
    img = Image.fromarray(img_array)
    
    # Save image to BytesIO buffer
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    
    # Encode as base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f'data:image/png;base64,{img_str}'
