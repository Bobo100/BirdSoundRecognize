# mp3轉檔wav
import os
from pydub import AudioSegment
import shutil

base_path = os.path.dirname(os.path.abspath(__file__))  #D:\Desktop\BirdCLEF-Baseline-master\My-Deep-Learning\dataset-download
mp3_dir =  os.path.join(base_path, "mp3") #D:\Desktop\BirdCLEF-Baseline-master\My-Deep-Learning\dataset-download\mp3
wav_dir = os.path.join(base_path, "wav") #D:\Desktop\BirdCLEF-Baseline-master\My-Deep-Learning\dataset-download\wav

# your folder path
input_mp3_folder = os.path.join(base_path, "dataset_input_folder_name_mp3")
output_wav_folder = os.path.join(base_path, "dataset_input_folder_name_wav")

covertfolder_mp3 = input_mp3_folder
covertfolder_wav = output_wav_folder

# 第一步 清除mp3和wav資料夾裡面 沒有聲音檔案的鳥聲資料夾
def remove_bird_folder():
    folders = list(os.walk(mp3_dir))[1:] #更改wav_dir train_dir test_dir 可以移除空白的資料夾
    for folder in folders:
    # folder example: ('FOLDER/3', [], ['file'])
        if not folder[2]:
            print(f'remove {(folder[0])}')
            os.rmdir(folder[0])
        print("done")
        
    folders = list(os.walk(wav_dir))[1:] #更改wav_dir train_dir test_dir 可以移除空白的資料夾
    for folder in folders:
    # folder example: ('FOLDER/3', [], ['file'])
        if not folder[2]:
            print(f'remove {(folder[0])}')
            os.rmdir(folder[0])
        print("done")
    print("remove done")

def mp3_to_wav():
    # 選擇走訪的資料夾 more20_mp3 // less20_mp3    
    for root, dirs, files in list(os.walk(covertfolder_mp3))[1:]: #mp3資料夾
        # print(root)
        dirname = root.split(os.path.sep)[-1]
        # print(dirname)
        # 根據 選擇的more20_mp3 對應 more20_wav less20_mp3 對應 less20_wav
        create_dir = os.path.join(covertfolder_wav, dirname)
        # print(create_dir)
        if not os.path.exists(create_dir):
            os.makedirs(create_dir)

        for name in files:
            # 計算還可以轉多少個檔案
            firstname, _, ext = name.rpartition(".")  #拆解檔名 XC187171.mp3
            # 如果附檔名等於mp3就作轉換
            if(ext == "mp3"):
                new_name = firstname + _ + "wav"      
                new_file_path = os.path.join("\\\?\\" + create_dir, new_name)
                if(not os.path.exists(new_file_path)):
                    try:
                        sound = AudioSegment.from_mp3(os.path.join("\\\?\\" + root, name))                    
                        print(f'sound duration_seconds = {sound.duration_seconds}')
                        # Change Frame Rate
                        # sound = sound.set_frame_rate(16000)
                        # Change Channel
                        # sound = sound.set_channels(1)
                        # Change Sample Width
                        # sound = sound.set_sample_width(2)
                        # 1 : “8 bit Signed Integer PCM”,
                        # 2 : “16 bit Signed Integer PCM”,
                        # 3 : “32 bit Signed Integer PCM”,
                        # 4 : “64 bit Signed Integer PCM” 
                        # Export the Audio to get the changed 
                        sound.export(new_file_path, format="wav")
                        print(new_file_path + " done")                    
                    except:
                        with open('error_message\error_mp3_to_wav_more20.txt', 'a') as f:
                                f.write(os.path.join("\\\?\\" + root, name))
                                f.write("\n")
                                f.close()
                        print("except")
                 
if __name__ == '__main__':
    print("\n==Export Start==")
    mp3_to_wav()
    print("done")
