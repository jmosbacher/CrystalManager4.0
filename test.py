from __future__ import absolute_import


from pyface.gui import GUI
from pyface.toolkit import toolkit_object

from pyface.ui.qt4.progress_dialog import ProgressDialog
from time import sleep
#ModalDialogTester = toolkit_object('util.modal_dialog_tester:ModalDialogTester')
#no_modal_dialog_tester = (ModalDialogTester.__name__ == 'Unimplemented')

class Test(object):
    def setUp(self):
        self.gui = GUI()
        self.dialog = ProgressDialog()

    def test(self):
        self.dialog.min = 0
        self.dialog.max = 10
        self.dialog.open()
        for i in range(11):
            result = self.dialog.update(i)
            self.gui.process_events()




"""
t = Thread(target=write_to_display,args= (display,))
t.setDaemon(True)
t.start()

display.configure_traits()

"""
a = Test()
a.setUp()
a.test()
sleep(3)