import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    # print(filepath)： /media/twil/Elements SE/sceneflow/

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    # print(classes)
    # ['driving__disparity', 'driving__frames_finalpass', 'flyingthings3d__disparity',
    # 'flyingthings3d__frames_finalpass', 'monkaa__disparity', 'monkaa__frames_finalpass']

    # find方法检测字符串中是否包含子字符串，如果包含子字符串返回开始的索引值，否则返回-1
    image = [img for img in classes if img.find('frames_finalpass') > -1]
    disp = [dsp for dsp in classes if dsp.find('disparity') > -1]
    # print(image)： ['driving__frames_finalpass', 'flyingthings3d__frames_finalpass', 'monkaa__frames_finalpass']
    # print(disp)： ['driving__disparity', 'flyingthings3d__disparity', 'monkaa__disparity']

    monkaa_path = filepath + [x for x in image if 'monkaa' in x][0] + "/frames_finalpass"
    monkaa_disp = filepath + [x for x in disp if 'monkaa' in x][0] + "/disparity"
    # print(monkaa_path)： /media/twil/Elements SE/sceneflow/monkaa__frames_finalpass/frames_finalpass
    # print(monkaa_disp)： /media/twil/Elements SE/sceneflow/monkaa__disparity/disparity

    monkaa_dir = os.listdir(monkaa_path)
    # print(monkaa_dir)：
    # ['a_rain_of_stones_x2', 'eating_camera2_x2', ... , 'treeflight_x2']

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    for dd in monkaa_dir:
        # print(dd)： a_rain_of_stones_x2
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
            all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    flying_path = filepath + [x for x in image if x == 'flyingthings3d__frames_finalpass'][0] + "/frames_finalpass"
    flying_disp = filepath + [x for x in disp if x == 'flyingthings3d__disparity'][0] + "/disparity"
    # print(flying_path)： /media/twil/Elements SE/sceneflow/flyingthings3d__frames_finalpass/frames_finalpass
    # print(flying_disp)： /media/twil/Elements SE/sceneflow/flyingthings3d__disparity/disparity

    flying_dir = flying_path+'/TRAIN/'
    subdir = ['A', 'B', 'C']

    for ss in subdir:
        # print(ss)： A
        flying = os.listdir(flying_dir+ss)
        # print(flying)：
        # ['0000', '0001', '0002', '0003', '0004', ... , '0048', '0049']

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
                all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    flying_dir = flying_path+'/TEST/'
    subdir = ['A', 'B', 'C']

    for ss in subdir:
        flying = os.listdir(flying_dir+ss)

        for ff in flying:
            imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
            for im in imm_l:
                if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                    test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)
          
                test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                    test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    driving_dir = filepath + [x for x in image if 'driving' in x][0] + '/frames_finalpass/'
    driving_disp = filepath + [x for x in disp if 'driving' in x][0] + "/disparity/"
    # print(driving_dir)： /media/twil/Elements SE/sceneflow/driving__frames_finalpass/frames_finalpass/
    # print(driving_disp)： /media/twil/Elements SE/sceneflow/driving__disparity/disparity/

    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.listdir(driving_dir+i+'/'+j+'/'+k+'/left/')
                for im in imm_l:
                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/left/'+im):
                        all_left_img.append(driving_dir+i+'/'+j+'/'+k+'/left/'+im)

                    all_left_disp.append(driving_disp+'/'+i+'/'+j+'/'+k+'/left/'+im.split(".")[0]+'.pfm')

                    if is_image_file(driving_dir+i+'/'+j+'/'+k+'/right/'+im):
                        all_right_img.append(driving_dir+i+'/'+j+'/'+k+'/right/'+im)

    # print(len(all_left_img), len(all_right_img), len(all_left_disp))： 35454 35454 35454
    # print(len(test_left_img), len(test_right_img), len(test_left_disp))： 4370 4370 4370
    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


