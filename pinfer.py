import onnxruntime as ort
import numpy as np
import cv2

def proprocess(filename):
    '''
    图片预处理流程
    '''
    image = cv2.imread(filename)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image, (512, 512))

    r = np.array(image_cv, dtype=float)
    img_np_n = r.flatten()
    print(img_np_n[:20])


    miu = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    img_np = np.array(image_cv, dtype=float) / 255.

    r = np.array(img_np, dtype=float)
    img_np_n = r.flatten()
    print(img_np_n[:20])
    

    r = np.array(image_cv, dtype=float)[:, :, 0]
    img_np_n = r.flatten()
    print(img_np_n[:10])

    r = (img_np[:, :, 0] - miu[0]) / std[0]
    img_np_n = r.flatten()
    print(img_np_n[:10])

    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b]).astype('float32')

    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

model = '/workspace/tmp/eff_isc_bs16_simplify.onnx'
img = '/workspace/tmp/orig.jpg'

ort_sess = ort.InferenceSession(model)

input_np = proprocess(img)

input_np_flat = input_np.copy()
input_np_flat = input_np_flat.flatten()
print(input_np_flat[:100])
c = np.insert(input_np, 0, values=input_np, axis=0)

for i in range(14):
    c = np.insert(c, 0, values=input_np, axis=0)
print(c.shape)
input_np = c


print(input_np.shape)

outputs = ort_sess.run(None, {'input': input_np})

# Print Result
result = outputs[0][0]
print(result)



# 51. 45. 43. 48. 48. 46. 47. 48. 50. 49
# -0.6      -0.64705884 -0.6627451  -0.62352943 -0.62352943 -0.6392157  -0.6313726  -0.62352943 -0.60784316 -0.6156863  -0.56078434 -0.6 -0.6784314  -0.73333335 -0.4745098   0.01176471 -0.78039217 -0.7254902
# -0.600000 -0.654902   -0.678431   -0.647059   -0.670588   -0.733333

# -0.6        -0.64705882 -0.6627451  -0.62352941 -0.62352941 -0.63921569

# 51	44	41	45	42	34	43	40 31	48	43	37	48	40	38	46 38	35	47	39
