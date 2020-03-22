import pandas as pd
import cv2
import os

# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
    i = 1
      
    for subdir, dirs, files in os.walk("E:/Project_Sem_6/batch2/"):    
        i = 1
        for filename in files:
            #print(subdir)
            #print(filename)
            dst =str(i) + ".jpg"
            src = subdir + "/" + filename 
            dst = subdir + "/" + dst 
              
            # rename() function will 
            # rename all the files 
            os.rename(src, dst) 
            i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 