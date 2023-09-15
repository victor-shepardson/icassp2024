import sys
import random
import argparse
import queue
import threading
import numpy as np
import numpy.typing as npt
import typing
from PySide6 import QtCore, QtWidgets, QtGui, QtCharts
import pyqtgraph as pg
from rtalign.gui import backend
from rtalign.gui import senders

import pythonosc
import pythonosc.osc_server


class DoubleSlider(QtWidgets.QSlider):
    """
    Continuous value QSlider that uses native double values instead of integers
    """

    # define a signal which enables Qt's callback system of signals/slots
    doubleValueChanged = QtCore.Signal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals
        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value:float):
        super(DoubleSlider, self).setValue(int(value * self._multi))


class AlignmentGraph(QtWidgets.QWidget):
    """
    Widget for displaying alignments
    """
    def __init__(self, parent:QtWidgets.QWidget=None):
        super().__init__(parent)

        pg.setConfigOption('imageAxisOrder', 'row-major') # best performance image data must be (height, width)
        #pg.setConfigOption('useNumba', True) # supposedly better performance for image data

        self.paint_enabled = False # set to True to enable alignment painting on the graph

        self.series = QtCharts.QLineSeries()
        self.series.setName("testdata")

        # Fill with some random monotonic alignment-looking data for demo purposes
        # pairs are: (time, encoded_token_idx)
        self.num_encodings = 40
        # self.imagedata = np.zeros((100,self.num_encodings), dtype=np.float32)
        # tok = 0
        # for t in range(100):
        #     frame = np.zeros(self.num_encodings, dtype=np.float32)
        #     frame[tok] = 1.0
        #     if tok < self.num_encodings:
        #         frame[tok+1] = 0.5
        #     if tok > 0:
        #         frame[tok-1] = 0.5

        #     self.imagedata[t] = frame

        #     if random.random() < 0.3:
        #         tok = tok+1

        self.frame = 0
        self.prev_tok = 0

        self.plot = pg.PlotWidget(parent=self)
        # See: https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html#pyqtgraph.ImageItem
        self.imageitem = pg.ImageItem()
        # self.imageitem.setImage(image=self.imagedata.T)
        self.plot.addItem(self.imageitem)
        self.plot.showAxes(True)
        self.plot.invertY(False) # vertical axis zero at the bottom

        self.attn_slider_resolution = 512
        self.attn_slider_max_value = (self.num_encodings-1) * self.attn_slider_resolution
        self.alignment_slider = QtWidgets.QSlider(parent=self)
        self.alignment_slider.setMinimum(0)
        self.alignment_slider.setMaximum( self.attn_slider_max_value )

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.plot)
        self.layout.addWidget(self.alignment_slider)
        self.setLayout(self.layout)

    def set_normalized_alignment(self, value:float):
        """
        Sets alignment slider to a normalized position from 0-1
        """
        self.alignment_slider.setValue(int(value * self.attn_slider_max_value))

    def set_alignment_as_token_idx(self, tok:int) -> None:
        """
        Set alignment slider to a position corresponding to a given token index
        between 0 and self.num_encodings-1

        tok     the target token index (must be between 0 and self.num_tokens-1)
        """

        if tok >=0 and tok < self.num_encodings:
            tok_as_sliderval = tok * self.attn_slider_resolution
            self.alignment_slider.setValue(tok_as_sliderval)
        else:
            print(f"Error: token index {tok} out of range (0-{self.num_encodings-1})")


    def get_slidervalue_as_embedding(self) -> np.ndarray:
    
        tok = self.alignment_slider.value() / self.attn_slider_resolution 

        # smoothing:
        tok = (tok + self.prev_tok) / 2
        tok_center = (tok + self.prev_tok)/2 # draw centered between (smoothed) position and prev position
        deltas = np.arange(self.num_encodings, dtype=np.float32) - tok_center
        deltas = deltas / (0.5 + abs(tok - self.prev_tok)) # sharpness modulated by speed 
        # discrete gaussian, sums exactly to 1 over text positions
        logits = -deltas**2
        res = np.exp(logits) / np.sum(np.exp(logits))

        self.prev_tok = tok
    
        return res[None]

    def addFrame(self, newframe:npt.ArrayLike):
        self.frame +=1
        self.imagedata = np.append(self.imagedata, newframe, axis=0)
        self.imageitem.setImage(image=self.imagedata.T)
        self.imageitem.update()

    def reset(self, num_encodings:int=None):
        """
        Reset the attention graph with a given number of encoded tokens (y-axis)
        Usually called after new text is encoded / attention recalculated.

        num_encodings sets the number of encoded tokens to scale the y-axis by
                        if not set the current number of encodings is left as-is
        """
        self.frame = 0
        if num_encodings is not None:
            self.num_encodings = num_encodings
        self.alignment_slider.setMinimum(0)
        self.alignment_slider.setMaximum(
            self.num_encodings * self.attn_slider_resolution)
        self.imagedata = np.zeros((1,self.num_encodings), dtype=np.float32)
        self.imageitem.setImage(image=self.imagedata.T)
        self.imageitem.update()
        #self.plot.update()
        

class RaveLatents(QtWidgets.QWidget):
    def __init__(self, 
                 parent:QtWidgets.QWidget=None, 
                 num_latents:int=8, pitch_slider:bool=False):
        super().__init__(parent)

        self.latents = list()
        self.layout = QtWidgets.QHBoxLayout(self)
        # latent widgets, each is a (SliderWidget, MeterWidget) pair
        # the first slide is bias, second is a display
        for idx in range(num_latents):
            if idx==0 and pitch_slider:
                bmin, bmax = -100., 100.
                vmin, vmax = 50., 550.
            else:
                bmin, bmax = -3., 3.
                vmin, vmax = -5., 5.

            bias_slider = DoubleSlider(decimals=3, parent=self)
            bias_slider.setMaximum(bmax)
            bias_slider.setMinimum(bmin)
            bias_slider.setValue(0.0)
            bias_slider.doubleValueChanged.connect(lambda val,latent=idx: self._bias_adjust(val, latent))
            
            value_meter = DoubleSlider(decimals=3, parent=self)
            value_meter.setMaximum(vmax)
            value_meter.setMinimum(vmin)
            value_meter.setValue(0.0)
            value_meter.setStyleSheet("""
                QSlider::groove:horizontal {
                    background-color: red;  /* Change the color here */
                    height: 10px;
                }
                
                QSlider::handle:horizontal {
                    background-color: blue;  /* Change the color here */
                    width: 20px;
                }
                """)

            self.latents.append((bias_slider, value_meter))
            self.layout.addWidget(bias_slider)
            self.layout.addWidget(value_meter)
            if idx < num_latents-1:
                self.layout.addSpacing(10)

    def _bias_adjust(self, val:float, latent:int):            
        # NOTE: The bias values of the sliders get sampled regularly in frame_callback, they are read by get_biases 
        print(f"ADJUST BIAS of LATENT {latent} = {val}")

    def get_bias(self, latent:int) -> float:
        return self.latents[latent][0].value()

    def get_biases(self) -> npt.NDArray[np.float32]:
        res = np.zeros((1,len(self.latents)), dtype=np.float32)
        for idx, (bias, _) in enumerate(self.latents):
            res[0][idx] = bias.value()
        return res
    
    def frame_callback(self, d):
        d['audio_t'] += self.get_biases()

    def set_latent_bias(self, latent:int, value:float):
        """
        Set a latent bias value in the GUI
        """
        self.latents[latent][0].setValue(value)

    def set_latents(self, values:typing.Union[list, npt.ArrayLike]):
        """
        Set latent values in the gui. 
        values > the number of sliders are ignored

        values  latent values as floats, in the shape (1, num_latents)
        """
        values = values[0] # trim off the extra dimension
        if len(values) <= len(self.latents):
            for idx,val in enumerate(values):
                self.latents[idx][1].setValue(val)
        else: # More values than sliders
            for idx,sliders in enumerate(self.latents):
                sliders[1].setValue(values[idx])


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, context:'MainWindow'=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")

        def osc_listen_button_callback(val):
            result=None
            addr = (self._oschostline.text(), int(self._oscportline.text()))
            status = self._osclistenbut.isChecked()
            if status:
                self._osclistenbut.setStyleSheet("color: black; background-color: lightblue")
                self._osclistenbut.setText("Listening..")
                # TODO: This use of MainWindow as context feels kludgy, should use an abstract class / mixin?
                if hasattr(context, "serve_osc"):
                    result = context.serve_osc(address=addr)
                else:
                    print("ERROR: No method 'serve_osc' defined on settings context.")
            else:
                self._osclistenbut.setStyleSheet("color: black; background-color: grey")
                self._osclistenbut.setText("Listen")
                if hasattr(context, "unserve_osc"):
                    result = context.unserve_osc()
                else:
                    print("ERROR: No method 'serve_osc' defined on settings context.")

            print(f"Listen button status: {status}")

        # Create GUI
        self._info = QtWidgets.QLabel("Something happened, is that OK?")

        self._osc = QtWidgets.QGroupBox(title="OSC Control Setup", parent=self)
        self._oschostline = QtWidgets.QLineEdit("localhost")
        self._oschostline.setMaxLength(17)
        self._oscportline = QtWidgets.QLineEdit("7777")
        self._oscportline.setMaxLength(5)
        self._osclistenbut = QtWidgets.QPushButton("&Listen")
        self._osclistenbut.setCheckable(True)
        self._osclistenbut.clicked.connect(osc_listen_button_callback)
        self._oscinfolabel = QtWidgets.QLabel("")
        hostportlayout = QtWidgets.QHBoxLayout()
        hostportlayout.addWidget(self._oschostline)
        hostportlayout.addWidget(QtWidgets.QLabel(":"))
        hostportlayout.addWidget(self._oscportline)
        hostport = QtWidgets.QWidget()
        hostport.setLayout(hostportlayout)
        osclayout = QtWidgets.QVBoxLayout()
        osclayout.addWidget(hostport)
        osclayout.addWidget(self._osclistenbut)
        osclayout.addWidget(self._oscinfolabel)
        self._osc.setLayout(osclayout)
        
        OKCLOSE = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self._buttonBox = QtWidgets.QDialogButtonBox(OKCLOSE)
        self._buttonBox.accepted.connect(self.accept)
        self._buttonBox.rejected.connect(self.reject)


        self._main_layout = QtWidgets.QVBoxLayout(self)
        self._main_layout.addWidget(self._info)
        self._main_layout.addWidget(self._osc)
        self._main_layout.addWidget(self._buttonBox)

        self.setLayout(self._main_layout)


class MainWindow(QtWidgets.QMainWindow):
    """
    Main Application Window
    """

    def __init__(self, 
        num_latents:int,
        parent:QtWidgets.QWidget=None, 
        use_backend:bool=True, 
        backend:backend.Backend=None, 
        sender:senders.Sender=None,
        update_fps:int=23
        ):
        super().__init__(parent)

        if sys.platform == 'darwin':
            self.setUnifiedTitleAndToolBarOnMac(True)

        self.mode = 'infer'
        
        self.update_fps = update_fps
        self.use_backend = use_backend
        self.backend = backend
        self.sender = sender
        if self.backend is None:
            print("No backend provided, disabling backend")
            self.use_backend = False
        # else:
            # self.backend.callback = sender
        
        # OSC
        self.osc_server = None

        # ---------------------------------------------------------------------
        # BUILD GUI
        # ---------------------------------------------------------------------
        self.setWindowTitle(QtCore.QCoreApplication.applicationName())
        self.main = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main)
        self.text_input = QtWidgets.QTextEdit(self.main)
        self.text_input.setMaximumHeight(100)
        self.text_input.setFontPointSize(24)
        self.text_input.setPlainText("It took me a while to find a voice, and now that I have one, I am not going to be silent.")
        self.btn_send_text = QtWidgets.QPushButton("Encode", parent=self.main)
        self.btn_send_text.clicked.connect(self.encode_text)
        self.text_encoding_status = QtWidgets.QLabel(self.main)
        self.text_encoding_status.setText("ʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ encoder feedback... ʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ")
        
        self.settings_dialog = None

        # App Toolbar
        self.toolbar = QtWidgets.QToolBar(self.main)
        self.addToolBar(self.toolbar)
        self._play_action = QtGui.QAction("Play", self)
        self._play_action.setStatusTip("Start Autoregression")
        self._play_action.triggered.connect(self.play_autoregression)
        self._pause_action = QtGui.QAction("Pause", self)
        self._pause_action.setStatusTip("Pause Autoregression")
        self._pause_action.triggered.connect(self.pause_autoregression)
        self._reset_action = QtGui.QAction("Reset", self)
        self._reset_action.setStatusTip("Reset Autoregression History")
        self._reset_action.triggered.connect(self.reset_autoregression)
        self._alignment_paint_toggle_action = QtGui.QAction("Attention Painting", self)
        self._alignment_paint_toggle_action.setCheckable(True)
        self._alignment_paint_toggle_action.setStatusTip("Toggle Attention Painting")
        self._alignment_paint_toggle_action.toggled.connect(self.toggle_alignment_paint)
        self._latent_feedback_toggle_action = QtGui.QAction("Latent Feedback", self)
        self._latent_feedback_toggle_action.setCheckable(True)
        self._latent_feedback_toggle_action.setStatusTip("Feed latent manipulation back to model")
        self._latent_feedback_toggle_action.toggled.connect(self.toggle_latent_feedback)
        self._settings_action = QtGui.QAction("Settings", self)
        self._settings_action.setStatusTip("Adjust OSC, MIDI and Audio settings")
        self._settings_action.triggered.connect(self.open_settings)
        self.temperature_slider = DoubleSlider(orientation=QtCore.Qt.Horizontal, decimals=3, parent=self)
        self.temperature_slider.setMaximum(5.0)
        self.temperature_slider.setMinimum(0.0)
        self.temperature_slider.setValue(1.0)
        self.temperature_slider.doubleValueChanged.connect(lambda val: self._temperature_adjust(val))

        self.toolbar.addAction(self._play_action)
        self.toolbar.addAction(self._pause_action)
        self.toolbar.addAction(self._reset_action)
        self.toolbar.addAction(self._alignment_paint_toggle_action)
        self.toolbar.addAction(self._latent_feedback_toggle_action)
        self.toolbar.addAction(self._settings_action)
        self.toolbar.addWidget(self.temperature_slider)

        # StatusBar
        self.statusbar = QtWidgets.QStatusBar(self.main)
        self.setStatusBar(self.statusbar)

        self.attention_graph = AlignmentGraph(self.main)

        self.latents = RaveLatents(
            self.main, num_latents=num_latents, pitch_slider=self.backend.use_pitch)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonFollowStyle)

        self._main_layout = QtWidgets.QVBoxLayout(self.main)
        self._main_layout.addWidget(self.toolbar)
        self._main_layout.addWidget(self.attention_graph)
        self._main_layout.addWidget(self.text_encoding_status)
        self._main_layout.addWidget(self.text_input)
        self._main_layout.addWidget(self.btn_send_text)
        self._main_layout.addWidget(self.attention_graph)
        self._main_layout.addWidget(self.latents)


        self.gui_update_timer = QtCore.QTimer(self)
        self.gui_update_timer.timeout.connect(self.update)
        
        # Used to generate fake data in update() when no backend is provided
        self.frame = 0
        self.tok = 30
        self.max_tok = 40
        self.gui_update_timer.start((1.0 / self.update_fps) * 1000)

    def update(self):
        """
        update method runs on a timer
        empties the queue from the backend and updates gui elements
        """
        self.frame += 1
        new_data = list()
        # empty message queue, update attention graph & RAVE latents
        if self.use_backend: 
            # it's fine to set this when in 'infer' mode, the backend
            # will ignore it
            attnval = self.attention_graph.get_slidervalue_as_embedding()
            self.backend.set_alignment(attnval)        

            for _ in range(self.backend.frontend_q.qsize()):
                try:
                    framedict = self.backend.frontend_q.get_nowait()
                    new_data.append(framedict)
                except queue.Empty as ex:
                    break
            
        else: # Do not use backend, instead generate random data.. useful for testing the gui (maybe?)
            new_attn_frame = np.zeros((1,self.attention_graph.num_encodings), dtype=np.float32)        
            new_latent_frame = np.random.rand(1,8) * 0.5
            new_latent_frame = new_latent_frame + self.latents.get_biases()
            if self.mode == 'infer':
                if random.random() < 0.2:
                    self.tok += random.choice([-1, 1])
            elif self.mode == 'paint':
                self.tok = self.attention_graph.alignment_slider.value()            
            else:
                raise ValueError(f"Unknown attention mode: {self.mode} - must be <infer|paint>")

            if self.tok >= self.max_tok-1:
                self.tok = self.max_tok - 1
            else:
                new_attn_frame[0,self.tok + 1] = 0.5
            if self.tok <= 0:
                self.tok = 0
            else:
                new_attn_frame[0,self.tok - 1] = 0.5
            new_attn_frame[0,self.tok] = 1.0

            new_data.append({'audio_t': new_latent_frame, 'align_t': new_attn_frame})
            
        # iterate through new_data and update the gui
        for datadict in new_data:
            if datadict.get('reset', False):
                self.finish_reset(num_tokens=datadict['align_t'].shape[-1])
            self.attention_graph.addFrame(datadict['align_t'])
        if len(new_data) > 0:
            self.latents.set_latents(values=new_data[-1]['audio_t'])

    def frame_callback(self, d):
        """
        Backend calls this after each `model.step`.
        Args:
            d: {'align_t':Tensor[batch, text len], 'audio_t':Tensor[batch, RAVE latent]}
        """
        # update frame from sliders
        self.latents.frame_callback(d)
        # send over OSC
        if self.sender is not None:
            self.sender(d)

    def finish_reset(self, num_tokens):
        """
        Called when backend signals a reset.
        """
        self.text_encoding_status.setText(f"ʭʬʭʬʭʬʭʬ encoded {num_tokens} embeddings ʬʭʬʭʬʭʬʭ")
        self.attention_graph.reset(num_encodings=num_tokens)

    def closeEvent(self, e:QtCore.QEvent):
        """
        Cleanup
        """
        # TODO: cleanup OSC/networking connections
        print(f"Application Close {e}")
        self.backend.cleanup()
        e.accept()
        #e.ignore() # Under some conditions ignore app close?

    def _temperature_adjust(self, temp=1.0) -> None:
        """
        Private method adjust model step inference temperature
        """
        if self.use_backend:
            self.backend.temperature = temp

    def encode_text(self):
        """
        Send input text to the text encoder backend
        """
        if self.use_backend:
            txtval = self.text_input.toPlainText()
            print(f"Encoding >{txtval}< ...")
            self.text_encoding_status.setText("ʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ encoding.... ʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ")
            self.num_embedding_tokens = self.backend.set_text(text=txtval)

        else:
            print("No backend enabled to encode text: ignoring...")

    def play_autoregression(self, val:bool):
        print(f"Play Autoregressive Frame Generator")
        if self.backend.text_t is None:
            self.encode_text()
        self.backend.start()

    def pause_autoregression(self, val:bool):
        print(f"Pause Autoregressive Frame Generator")
        self.backend.pause()

    def reset_autoregression(self, val:bool):
        print(f"Reset autoregression history")
        # self.attention_graph.reset()
        self.backend.reset()

    def set_temperature(self, temp:float) -> None:
        """
        Set inference temperature (used by OSC/MIDI)

        Args:
            temp  inference temperature from 0.0-5.0
        """
        if temp > 5.0:
            temp=5.0
        elif temp < 0:
            temp=0
        self.temperature_slider.setValue(temp)

    def toggle_alignment_paint(self, toggle:bool):
        if toggle:
            self.mode = 'paint'
        else:
            self.mode = 'infer'
        self.backend.set_mode(self.mode)
        print(f"Alignment Mode Changed To:{self.mode}")

        
    def toggle_latent_feedback(self, toggle:bool):
        self.backend.set_latent_feedback(toggle)
        print(f"Latent Feedback Status Changed To:{toggle}")

    def set_alignment_mode(self, mode:str='infer') -> None:
        """
        Set alignment mode directly (used by OSC/MIDI)
        """
        if mode in ['infer', 'paint']:
            self.mode = mode
            self.backend.set_mode(self.mode)
            self._alignment_paint_toggle_action.setChecked((mode == 'paint'))

    def open_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(context=self)
        self.settings_dialog.show() # use show() to display a modeless dialog

    def unserve_osc(self):
        self.osc_server.shutdown()
        self.osc_server.server_close()
        self.osc_server = None
        print("Waiting for server to shutdown...")
        self.osc_server_thread.join()
        print("Server Thread closed...")


    def serve_osc(self, address=("localhost", 7777)):

        if self.osc_server is not None:
            self.unserve_osc()

        def osc_unknown(addr:str, *args:list[typing.Any]) -> None:
            print(f"Unknown OSC address: {addr}  with: '{args}'")

        def osc_play(addr:str, *args:list[typing.Any]) -> None:
            self._play_action.trigger()

        def osc_pause(addr:str, *args:list[typing.Any]) -> None:
            self._pause_action.trigger()

        def osc_reset(addr:str, *args:list[typing.Any]) -> None:
            self._reset_action.trigger()

        def osc_alignment_mode(addr:str, mode:str) -> None:
            self.set_alignment_mode(mode)

        def osc_latent_feedback(addr:str, val:bool) -> None:
            self._latent_feedback_toggle_action.setChecked(val)

        def osc_set_text(addr:str, text:str, encode:bool=True) -> None:
            self.text_input.setText(text)
            if encode:
                self.btn_send_text.click()

        def osc_set_alignment_as_token_idx(addr:str, tok_idx:int, force_paint:bool=False) -> None:
            self.attention_graph.set_alignment_as_token_idx(tok_idx)

        def osc_set_alignment_normalized(addr:str, normalized_align:float, force_paint:bool=False) -> None:
            self.attention_graph.set_normalized_alignment(normalized_align)

        def osc_set_bias(addr:str, latent:int, bias:float) -> None:
            self.latents.set_latent_bias(latent, bias)
            print(f"Set latent {latent} bias: {bias}")

        def osc_set_temperature(addr:str, temp:float) -> None:
            self.set_temperature(temp)
            print(f"Set sampling temp {temp}")

        self.osc_dispatcher = pythonosc.dispatcher.Dispatcher()
        self.osc_dispatcher.set_default_handler(osc_unknown)
        # OSC Callback mappings
        self.osc_dispatcher.map("/play", osc_play)
        self.osc_dispatcher.map("/pause", osc_pause)
        self.osc_dispatcher.map("/reset", osc_reset)
        self.osc_dispatcher.map("/alignment_mode", osc_alignment_mode)
        self.osc_dispatcher.map("/latent_feedback", osc_latent_feedback)
        self.osc_dispatcher.map("/set_text", osc_set_text)
        self.osc_dispatcher.map("/set_token", osc_set_alignment_as_token_idx)
        self.osc_dispatcher.map("/set_alignment", osc_set_alignment_normalized)
        self.osc_dispatcher.map("/set_bias", osc_set_bias)
        self.osc_dispatcher.map("/set_temperature", osc_set_temperature)

        def run_osc_server(context):
            print(f"Serving on {context.osc_server.server_address}")
            context.osc_server.serve_forever()
            print(f"Closing OSC Server...")

        self.osc_server = pythonosc.osc_server.ThreadingOSCUDPServer( address, self.osc_dispatcher )
        self.osc_server_thread = threading.Thread(target=run_osc_server, args=(self,), daemon=True)
        self.osc_server_thread.start()
        print("OSC Server Thread Started")




if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        description='rtVoice App', 
        formatter_class=argparse.RawTextHelpFormatter)
    
    # TODO: Probably want to pass in some OSC config and maybe other stuff...
    # TODO: port argparsing over to fire? I like that argparse is so explicit, but maybe I'm just old
    parser.add_argument("--ckpt", help="TTS model checkpoint file", type=pathlib.Path, default=None, required=True)
    parser.add_argument("--rave-path", help="RAVE vocoder torchscript file. If provided the script will run in stand-alone audio mode, without depending on SuperCollider.", type=pathlib.Path, default=None, required=False)
    parser.add_argument("--audio-out", help="audio output device name or index", type=str, default=None, required=False)
    parser.add_argument("--audio-block", help="audio block size", type=int, default=None, required=False)
    parser.add_argument("--send-scsynth", help="Send latents to scsynth via OSC. (by default sends to sclang)", default=False, action='store_true')
    parser.add_argument("--no-backend", help="Do not enable backend. Potentially useful for testing the gui.", default=False, action='store_true')
    args = parser.parse_args()

    if args.audio_out == 'default':
        args.audio_out = None
    if args.audio_out is not None and args.audio_out.isdecimal():
        args.audio_out = int(args.audio_out)

    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.setOrganizationName("Intelligent Instruments Lab")
    QtCore.QCoreApplication.setApplicationName("rtVoice")
    QtCore.QCoreApplication.setApplicationVersion(QtCore.qVersion())

    if args.rave_path: # Use stand-alone audio mode. 
        sender = None
    elif args.send_scsynth: # Send latents to scsynth via OSC
        sender = senders.SCSynthDirectOSCSender(
            host='127.0.0.1',
            port=57110,
            bus_index=64,
            latency=0.2
        )
    else: # Send latents to sclang via OSC
        sender = senders.GenericOSCSender(
            host='127.0.0.1', 
            port=57120, 
            lroute='/rtalign/latents',
            sroute='/rtalign/status'
        )

    backend = backend.Backend(
        checkpoint=args.ckpt, 
        rave_path=args.rave_path,
        audio_out=args.audio_out,
        audio_block=args.audio_block
        )
    win = MainWindow(
        use_backend=(not args.no_backend), 
        sender=sender,
        backend=backend,
        num_latents=backend.num_latents)
    backend.frame_callback = win.frame_callback

    available_geometry = win.screen().availableGeometry()
    win.resize(available_geometry.width() / 2, available_geometry.height())
    win.move((available_geometry.width() - win.width()) / 2, 0)
    win.show()
    sys.exit(app.exec())