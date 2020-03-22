from PIL import Image
import numpy as np
import os
import csv

'''list_row = []
for i in range(0,10000):
    list_row.append("C" + str(i+1))
print(list_row[0])'''

'''for subdir, dirs, files in os.walk("E:/Project_Sem_6/batch1/"):    
    for filename in files:
        file = subdir + ".csv"
        #print(file)
        with open(file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list_row)'''

for subdir, dirs, files in os.walk("E:/Project_Sem_6/batch1/"):    
        for filename in files:
            #print(subdir)
            img_file = Image.open(os.path.join(subdir, filename))
            #img_file.show()
        
            # get original image parameters...
            width, height = img_file.size
            format = img_file.format
            mode = img_file.mode
        
            newsize = (160,160) 
            img_file1 = img_file.resize(newsize) 
        
            # Make image Greyscale
            #img_grey = img_file1.convert('L')
            #img_grey.save('result.png')
            #img_grey.show()
            #print(os.path.basename(subdir))
            # Save Greyscale values
            value = np.asarray(img_file1.getdata(), dtype=np.int).reshape((img_file1.size[1], img_file1.size[0]))
            value = value.flatten()
            #print(value)
            file = "E:/Project_Sem_6/CSV_file_160/" + os.path.basename(subdir) + ".csv"
            #print(file)
            '''if os.path.isfile(file):
                print (file + "File exist")
            else:
                print (file + "File not exist")'''
            with open(file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value)

for subdir, dirs, files in os.walk("E:/Project_Sem_6/batch2/"):    
        for filename in files:
            #print(subdir)
            img_file = Image.open(os.path.join(subdir, filename))
            #img_file.show()
        
            # get original image parameters...
            width, height = img_file.size
            format = img_file.format
            mode = img_file.mode
        
            newsize = (160,160) 
            img_file1 = img_file.resize(newsize) 
        
            # Make image Greyscale
            #img_grey = img_file1.convert('L')
            #img_grey.save('result.png')
            #img_grey.show()
            #print(os.path.basename(subdir))
            # Save Greyscale values
            value = np.asarray(img_file1.getdata(), dtype=np.int).reshape((img_file1.size[1], img_file1.size[0]))
            value = value.flatten()
            #print(value)
            file = "E:/Project_Sem_6/CSV_file_160/" + os.path.basename(subdir) + ".csv"
            #print(file)
            '''if os.path.isfile(file):
                print (file + "File exist")
            else:
                print (file + "File not exist")'''
            with open(file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(value)
