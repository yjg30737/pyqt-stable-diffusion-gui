from qtpy.QtCore import QThread, Signal

from src.script import generate_image, image_to_grid, save_image


class Thread(QThread):
    generateFinished = Signal(list)
    generateFailed = Signal(str)

    def __init__(self, pipeline, generation_count, model_id, save_path, rows, cols, **pipeline_args):
        super(Thread, self).__init__()
        self.__pipeline = pipeline
        self.__generation_count = generation_count
        self.__model_id = model_id
        self.__save_path = save_path
        self.__rows = rows
        self.__cols = cols
        self.__pipeline_args = pipeline_args

    def __generate_save_image(self):
        try:
            images = generate_image(self.__pipeline, **self.__pipeline_args)
            prompt = self.__pipeline_args['prompt']
            filename = ''
            if len(images) > 1:
                grid = image_to_grid(images, rows=self.__rows, cols=self.__cols)
                suffix = f'({self.__rows}x{self.__cols} grid)'
                filename = save_image([grid], prompt=prompt, model_id=self.__model_id, save_path=self.__save_path, suffix=suffix)
                # have to put upscale code, i can't test it because of OutOfMemoryError
            else:
                filename = save_image(images, prompt=prompt, save_path=self.__save_path, model_id=self.__model_id)
            return filename
        except Exception as e:
            raise Exception(e)

    def run(self):
        try:
            filenames = []
            if self.__generation_count == -1:
                while True:
                    filename = self.__generate_save_image()
                    filenames.append(filename)
            else:
                for i in range(self.__generation_count):
                    filename = self.__generate_save_image()
                    filenames.append(filename)
            self.generateFinished.emit(filenames)
        except Exception as e:
            self.generateFailed.emit(str(e))
