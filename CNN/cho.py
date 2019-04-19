import wx
import win32api
from CNN.load_meta_cnn import *
data_folder_name = './temp'
if not os.path.exists(data_folder_name):
    os.mkdir(data_folder_name)


class Config(object):
    app_name = 'Emotion Classifier'
    app_border = 3
    app_size = (320, 600)
    file_name = []
    file_path = []
    flag = False


class TextEditor(wx.App):
    def OnInit(self):
        return True

    def create_frame(self, config=None):
        if config is None:
            self.config = Config()
        else:
            self.config = config
        self.frame = Frame(None, title=self.config.app_name, size=self.config.app_size, content=self.config)
        self.frame.Show()


class Frame(wx.Frame):
    def __init__(self, parent=None, id=-1, title='',
                 pos=wx.DefaultSize, size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE, content=None):
        wx.Frame.__init__(self, parent=parent, title=title, size=size)
        self.InitUI(content)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, evt):
        self.Destroy()

    def InitUI(self, con):

        self.config = con
        self.panel = wx.Panel(self)
        h_box = wx.BoxSizer()
        h_box_content = wx.BoxSizer()
        v_box = wx.BoxSizer(wx.VERTICAL)
        self.v_box_l = wx.BoxSizer(wx.VERTICAL)
        self.v_box_r = wx.BoxSizer(wx.VERTICAL)
        self.button_open = {}
        self.button_choose_file = wx.Button(self.panel, label="选择图片")
        self.button_reset = wx.Button(self.panel, label="重置")
        self.button_open_camera = wx.Button(self.panel, label="开启摄像头")
        self.text_content_file = wx.TextCtrl(self.panel, value='', style=wx.TE_MULTILINE | wx.TE_RICH2
                                                                         | wx.TE_PROCESS_ENTER | wx.TE_READONLY)

        h_box.Add(self.button_choose_file, proportion=0, flag=wx.LEFT | wx.ALL, border=self.config.app_border)
        h_box.Add(self.button_open_camera, proportion=0, flag=wx.LEFT | wx.ALL, border=self.config.app_border)
        h_box.Add(self.button_reset, proportion=0, flag=wx.LEFT | wx.ALL, border=self.config.app_border)
        v_box.Add(h_box, proportion=0, flag=wx.EXPAND | wx.ALL, border=self.config.app_border)
        v_box.Add(self.text_content_file, proportion=1, flag=wx.EXPAND | wx.ALL, border=self.config.app_border)
        self.panel.SetSizer(v_box)

        self.button_choose_file.Bind(wx.EVT_BUTTON, self.action_choose_file)
        self.button_open_camera.Bind(wx.EVT_BUTTON, self.action_open_camera)
        self.button_reset.Bind(wx.EVT_BUTTON, self.action_reset_re)

        # self.SetMenuBar(menu_bar)
        exeName = win32api.GetModuleFileName(win32api.GetModuleHandle(None))
        icon = wx.Icon(exeName, wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)
        # for k, v in self.config.button_dict_rev.items():
        #     self.action_reborn(k, v)
        return True

    def action_choose_file(self, evt):
        files_filter = "Text (*.txt)" "All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self,
            message="Open",
            wildcard=files_filter,
            style=wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            file_path = dlg.GetPaths()
            self.config.file_name += dlg.Filenames
        else:
            return
        self.text_content_file.SetValue('')
        for name_ in self.config.file_name:
            self.text_content_file.AppendText('{}\n'.format(name_))
        for i, file_p in enumerate(file_path):
            if file_p.lower().endswith('jpg') or file_p.endswith('png'):
                if not show_img_after_classifier(file_p, i):
                    self.show_message("该图片中未检测到人脸!")

    def action_open_camera(self, evt):
        if not actual_time_classifier():
            self.show_message("没有检测到摄像头!")

    def action_open(self, evt):
        name = evt.GetEventObject().GetLabel()
        path = self.config.button_dict_rev[name]
        if not self.config.del_flag:
            win32api.ShellExecute(0, 'open', path, '', '', 1)
        else:
            self.button_open[path].Destroy()
            del self.button_open[path]
            del self.config.button_dict_rev[name]

    def show_message(self, message):
        wx.MessageBox(message, "提示", wx.OK | wx.ICON_INFORMATION)

    def action_reset_re(self, evt):
        self.config.file_path = []
        self.config.file_name = []
        self.text_content_file.SetValue('')


if __name__ == '__main__':
    text_app = TextEditor()
    text_app.create_frame()
    text_app.MainLoop()

