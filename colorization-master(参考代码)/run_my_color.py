import matplotlib.pyplot as plt  
from colorizers import eccv16, siggraph17  
from colorizers.util import load_img, preprocess_img, postprocess_tens  
  
# 1. 加载模型 (会自动下载权重)  
# eccv16 是论文原始版本，siggraph17 是后来的改进版(通常效果更自然)  
colorizer_eccv16 = eccv16(pretrained=True).eval()  
colorizer_siggraph17 = siggraph17(pretrained=True).eval()  
  
# 2. 读取并预处理图片  
# 请将 'imgs/test.jpg' 换成你自己的图片路径  
img = load_img('imgs/ansel_adams3.jpg')   
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))  
  
# 如果你有 GPU，可以取消下面的注释加速  
# import torch  
# colorizer_eccv16.cuda()  
# colorizer_siggraph17.cuda()  
# tens_l_rs = tens_l_rs.cuda()  
  
# 3. 进行上色推理  
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())  
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())  
  
# 4. 显示或保存结果  
plt.figure(figsize=(12,8))  
plt.subplot(1,3,1)  
plt.imshow(img)  
plt.title('Original')  
plt.axis('off')  
  
plt.subplot(1,3,2)  
plt.imshow(out_img_eccv16)  
plt.title('ECCV16 Output')  
plt.axis('off')  
  
plt.subplot(1,3,3)  
plt.imshow(out_img_siggraph17)  
plt.title('SIGGRAPH17 Output')  
plt.axis('off')  
  
plt.show()  
  
# 如果要保存图片：  
# plt.imsave('output_eccv16.png', out_img_eccv16)  
