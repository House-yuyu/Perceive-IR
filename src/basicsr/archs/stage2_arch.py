
from collections import OrderedDict

import clip
import torch
from torchvision.transforms import Compose, Resize, Normalize

from basicsr.archs.restormer_arch import *
from basicsr.archs.restormer_arch import TransformerBlock as RestormerBlock
from einops import einsum, rearrange
from torch import einsum

############# Rcp #################
def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    # res_channel = []
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

### -- LKA --- ###
### from https://github.com/AlexZou14/CVHSSR/blob/main/basicsr/models/archs/CVHSSR_arch.py
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
### -- CHIMB -- ###
### from https://github.com/AlexZou14/CVHSSR/blob/main/basicsr/models/archs/CVHSSR_arch.py
class CHIMB(nn.Module):  # GFDN
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca1 = LKA(dw_channel // 2)
        self.sca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, stride=1, padding=1,
                               groups=ffn_channel, bias=True)
        self.conv6 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca2(x) * x + self.sca1(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x1, x2 = self.conv5(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv6(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
## SRiR-CHIMB
class RIR_CHIMB(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR_CHIMB, self).__init__()
        module_body = [
            # RB(n_feats) for _ in range(n_blocks)
            CHIMB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)   # SE_ResBlock  - SE_ResBlock - Conv
        res += x
        return self.relu(res)
    
class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)
    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.relu(x)
        return x
    
class res_ch_CHIMB(nn.Module):
    def __init__(self, n_feats, out_ch=None, blocks=2):
        super(res_ch_CHIMB,self).__init__()
        self.match_channel = out_ch

        self.conv_init1 = convd(3, n_feats//2, 3, 1)
        self.conv_init2 = convd(n_feats//2, n_feats, 3, 1)
        # self.extra = RIR(n_feats, n_blocks=blocks)
        self.extra = RIR_CHIMB(n_feats, n_blocks=blocks)

        if out_ch!=None:
            self.conv_out = convd(n_feats, out_ch, 3, 1)       

    def forward(self,x):
    
        x = self.conv_init2(self.conv_init1(x))
        x = self.extra(x)
        if self.match_channel!=None:
            x = self.conv_out(x)    
        return x
    
############# Decoder #################
class DecoderBlock(nn.Module):
    def __init__(self,
                 in_dim, heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                 prompt_dim = 512, rcp_dim= 48):
        super(DecoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.cross_att_text = CrossAttention_LDM(query_dim=in_dim, context_dim=prompt_dim)        
        self.cross_att_img = CrossAttention_LDM(query_dim=in_dim, context_dim=prompt_dim)         
        self.fuse_rcp = Modulation2D(inChannels=in_dim)
        self.RestormerBlock = nn.Sequential(*[RestormerBlock(dim=in_dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

    def forward(self, x, text_emb, rcp_feature, hq_emb, iter):

        b, c, h, w = x.shape

        if iter==0:
            x = rearrange(x, 'b c h w -> b (h w) c')
            if text_emb.ndim==2:
                text_emb = text_emb.unsqueeze(0)                              
            x = self.cross_att_text(self.norm1(x), text_emb) + x              
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            if hq_emb.ndim==2:
                hq_emb = hq_emb.unsqueeze(0)                             
            x = self.cross_att_img(self.norm2(x), hq_emb) + x              
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            gamma, beta =  self.fuse_rcp(rcp_feature)
            x = x + gamma*x  + beta


        h = self.RestormerBlock(x)
        return h

# LDM cross attention 
class CrossAttention_LDM(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):      
        h = self.heads

        q = self.to_q(x)
        # context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale       
        attn = sim.softmax(dim=-1)        

        out = einsum('b i j, b j d -> b i d', attn, v)                  
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

    
# Modulation2D
class Modulation2D(nn.Module):
    def __init__(self, inChannels):
        super(Modulation2D, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1, stride=1)

        self.conv_gama = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        self.conv_beta = nn.Conv2d(inChannels, inChannels, 1, padding=0, stride=1)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))

        gama = self.sigmoid(self.conv_gama(out))       
        beta = self.conv_beta(out)

        return gama, beta
    

class cyclicPrompt(nn.Module):
    def __init__(self,
                 inp_channels=3,        
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,      ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 model_clip = None,      
                 num_vector = 8,
                 iter_times = 2          
                 ):
        super(cyclicPrompt, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()

        self.iter_times = iter_times

        # 
        self.num_vector = num_vector
        if self.num_vector != 0 :
            learnable_vector = torch.empty(self.num_vector, 512 )
            nn.init.normal_(learnable_vector, std=0.02)             # 初始化
            self.learnable_vector = nn.Parameter(learnable_vector)
            print(f'learnable context length: "{self.learnable_vector.shape}"')
        else:
            print(f'Do NOT use learnable lq-aware vector.')

        if self.num_vector != 0 :
            self.deg_aware = nn.Sequential(OrderedDict([
                ('linear1',nn.Linear(512, 512//16)),
                ('layernorm',nn.LayerNorm(512//16)),
                ('relu1',nn.ReLU(inplace=True)),
                ('linear2',nn.Linear(512//16, 512)),
            ]))

        self.clip_input_preprocess = Compose([
            Resize([224, 224]),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

        ###############################################################
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        ###############################################################
        # Decoder
        self.up4_3 = Upsample(int(dim * 2 ** 3))                                ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        # modify Decoder Block Here
        self.decoder_level3 =DecoderBlock(in_dim=dim * 2 ** 2, heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                     LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2]//2)        ## Decoder部分Block数量减半

        self.up3_2 = Upsample(int(dim * 2 ** 2))                               ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = DecoderBlock(in_dim=dim * 2 ** 1, heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                     LayerNorm_type=LayerNorm_type,num_blocks=num_blocks[1]//2)


        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = DecoderBlock(in_dim=dim * 2 ** 1, heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                     LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0]//2)

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.rcp_extractor = res_ch_CHIMB(n_feats=dim)      
        self.rcp_down1 = Downsample(int(dim))                 
        self.rcp_down2 = Downsample(int(dim*2))               
        self.rcp_ch = nn.Conv2d(dim, dim*2, 3, padding="same")        
 
        self.hq_mlp = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(512, 512)),
            ('layernorm',nn.LayerNorm(512)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(512, 512)),
            ('layernorm',nn.LayerNorm(512)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(512, 512)),
            ]))

    def reset_state(self):
        self.should_ret = True         

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature
    
    @torch.no_grad()
    def get_img_feature(self, img):
        img_input_clip = self.clip_input_preprocess(img)
        img_feature = self.model_clip.encode_image(img_input_clip)
        return img_feature
    
    def forward(self, imgs, texts):

        self.reset_state()

        # 输入文本CLIP特征
        text_feat = self.get_text_feature(texts)
        # 输入LQ图像CLIP特征
        lq_clip_feat = self.get_img_feature(imgs)

        # (8,512) + (b,512) : (1, 8, 512) + (b, reapt 8, 512)
        if self.num_vector != 0 :
            degware_vector = self.learnable_vector.repeat(lq_clip_feat.shape[0],1,1) + self.deg_aware(lq_clip_feat).unsqueeze(1).repeat(1,self.num_vector,1)
            prompt = torch.concat([degware_vector, text_feat.unsqueeze(1)], dim=1)    
        else:
            prompt = text_feat
            
        # Encoder
        inp_enc_level1 = self.patch_embed(imgs)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_li = []
        for i in range(self.iter_times):
            if self.should_ret:
                hq_clip_feat = None
                rcp_feature = None
                rcp_level3 = None
                rcp_level2 = None
                rcp_level1 = None

            out_dec_level3 = self.decoder_level3(inp_dec_level3, prompt, rcp_level3, hq_clip_feat, iter = i)    # dim=192

            inp_dec_level2 = self.up3_2(out_dec_level3)             # B,dim*2**1,H//2,W//2
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2, prompt, rcp_level2, hq_clip_feat,  iter = i)    # dim=96

            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)                                     # dim=96
            out_dec_level1 = self.decoder_level1(inp_dec_level1, prompt, rcp_level1, hq_clip_feat, iter = i)

            out_dec_level1 = self.refinement(out_dec_level1)

            #### For Dual-Pixel Defocus Deblurring Task ####
            if self.dual_pixel_task:
                out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
                out_dec_level1 = self.output(out_dec_level1)
            else:
                out_dec_level1 = self.output(out_dec_level1) + imgs

            if self.should_ret:
                ch_res_out = get_residue(out_dec_level1)
                rcp_feature = self.rcp_extractor(torch.cat([ch_res_out, ch_res_out, ch_res_out], dim=1)) 
                rcp_level1 = self.rcp_ch(rcp_feature)          
                rcp_level2 = self.rcp_down1(rcp_feature)       
                rcp_level3 = self.rcp_down2(rcp_level2)        

                hq_clip_feat = self.get_img_feature(out_dec_level1)
                hq_clip_feat = self.hq_mlp(hq_clip_feat)
                hq_clip_feat = torch.concat([hq_clip_feat.unsqueeze(1), text_feat.unsqueeze(1)], dim=1) # (b,2,512)

                self.should_ret = False

            out_li.append(out_dec_level1)

        out = torch.concatenate(out_li, dim=0).mean(dim=0, keepdim=True)     

        return out, out_li


if __name__ =='__main__':

    from clip import clip
    model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
    model.float()
    for para in model.parameters():
        para.requires_grad = False

    net = cyclicPrompt(model_clip=model, num_vector=8, num_blocks=[ 4,6,6,8 ], iter_times=2,)

    img_shape = (1, 3, 256, 256)
    input_img = torch.randn(img_shape)
    tokenized_text = torch.randint(0, 3000, (img_shape[0], 77), dtype=torch.int32)
    
    from thop import profile
    print("---- model complexity evaluate by thop profile----")
    flops, params = profile(net, inputs=(input_img, tokenized_text), report_missing=True)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))            # 
    print("params=", str(params / 1e6) + '{}'.format("M"))          # 
    print('\n')

    print("---- model complexity evaluate by thop profile----")
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"), end='\t')      # 230.42 G
    print("params=", str(params / 1e6) + '{}'.format("M"))              # 


    print("---- model complexity evaluate by customized code ----")
    n_param = sum([p.nelement() for p in net.parameters()])             # 所有参数数量
    n_param_train = sum([p.nelement() for p in net.parameters() if p.requires_grad])  # 只计算参与更新的参数数量
    print('Total params:', str(n_param / 1e6) + '{}'.format("M"))
    print('Tranable params:', str(n_param_train / 1e6) + '{}'.format("M"))
    print('\n')

    out, out_li = net(input_img, tokenized_text)
    print(out.shape)