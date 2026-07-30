import logging
import os
import time

class Log:
    def __init__(self,save_path):
        '''
        初始化函数
        输入：
            save_path:日志保存的路径
        '''
        print('Log init')
        self.isactivate = False
        self.save_path = save_path
        self.count = 0

    def open(self):
        '''
        打开日志文件的方法，遍历指定路径下的文件夹，找到下一个可用的实验文件夹（命名为'exp'+数字），
        如果已存在，则取最大数字加1
        创建一个以时间命名的文本文件，并将其作为日志文件进行写入
        '''
        dir_list = os.listdir(self.save_path)
        self.count = 0
        while 'exp' + str(self.count) in dir_list:
            self.count += 1
        os.makedirs(os.path.join(self.save_path,'exp'+str(self.count)),exist_ok=True)
        self.save_path = os.path.join(self.save_path,'exp'+str(self.count))
        time_str = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime())
        file_name = os.path.join(self.save_path, time_str + '.txt')
        self.f = open(file_name,'w')
        print('open file success')

    def activate(self):
        self.isactivate = True

    def deactivate(self):
        self.isactivate = False

    def log(self,*args):
        '''
        将日志信息写入文件中
        '''
        txt = ''.join(str(it) for it in args)
        print(txt)
        if self.isactivate:
            self.f.write(txt)
            self.f.write('\n')

    def get_save_path(self,file_name):
        '''
        返回文件的完整保存路径
        '''
        return os.path.join(self.save_path,file_name)

    def read_save_path(self,file_name):
        '''
        从日志中读取文件路径
        '''
        tmp_list = self.save_path.split('\\')[:-1]
        tmp_path = '\\'.join(i for i in tmp_list)
        dir_list = os.listdir(tmp_path)
        count = 0
        while 'exp' + str(count) in dir_list:
            count += 1
        count -= 1
        exist_flag = False
        while not exist_flag:
            save_path = os.path.join(tmp_path,'exp'+str(count))
            for files in os.listdir(save_path):
                ext = files.spllit('.')[-1]
                if ext == 'pt':
                    exist_flag = True
                    break
            count -= 1
        return os.path.join(save_path,file_name)     

    def close(self):
        self.f.close()
        print('close file success')

    def delete(self):
        for files in os.listdir(self.save_path):
            os.remove(os.path.join(self.save_path,files))
        os.removedirs(self.save_path)
        print('Remove dirs success')

if __name__ == '__main__':
    l = Log(r'E:\AI_anke\MyCode\logs')
    l.open()
    l.close()
    l.delete()
