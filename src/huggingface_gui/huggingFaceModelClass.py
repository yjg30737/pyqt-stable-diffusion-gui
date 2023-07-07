import importlib

from huggingface_hub import scan_cache_dir, RepoCard
from transformers import AutoConfig, TRANSFORMERS_CACHE


def format_size(num: int) -> str:
    """Format size in bytes into a human-readable string.

    Taken from https://stackoverflow.com/a/1094933
    """
    num_f = float(num)
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num_f) < 1000.0:
            return f"{num_f:3.1f}{unit}"
        num_f /= 1000.0
    return f"{num_f:.1f}Y"


class HuggingFaceModelClass:
    """
    to search HuggingFaceModel information
    """
    def __init__(self):
        super(HuggingFaceModelClass, self).__init__()
        self.__initVal()

    def __initVal(self):
        self.__cache_dir = TRANSFORMERS_CACHE
        self.__text_2_image_only = False

    def setCacheDir(self, cache_dir):
        self.__cache_dir = cache_dir

    def setText2ImageOnly(self, f: bool):
        """
        this is only applied in "getModels" and "getModelsSize"
        :param f:
        :return:
        """
        self.__text_2_image_only = f

    def getModels(self, certain_models=None):
        models = [{"id": i.repo_id, "size_on_disk": i.size_on_disk, "size_on_disk_str": i.size_on_disk_str,
                   "is_t2i": True if 'text-to-image' in
                                      RepoCard.load(i.repo_id).data.get('tags', []) else False, }
                    for i in scan_cache_dir(cache_dir=self.__cache_dir).repos]
        if self.__text_2_image_only:
            models = [model for model in models if model['is_t2i']]
        if certain_models is None:
            return models
        else:
            return list(filter(lambda x: x['id'] in certain_models, models)) if len(
                certain_models) > 0 else certain_models

    def getModelsSize(self, certain_models=None):
        models_size = scan_cache_dir(cache_dir=self.__cache_dir).size_on_disk_str
        if self.__text_2_image_only:
            text_2_image_only_size = sum(list(map(lambda x: x['size_on_disk'], filter(lambda x: x['is_t2i'], self.getModels(certain_models)))))
            return format_size(text_2_image_only_size)
        if certain_models is None:
            return models_size
        else:
            certain_models_size = sum(list(map(lambda x: x['size_on_disk'], self.getModels(certain_models))))
            return format_size(certain_models_size)

    def installHuggingFaceModel(self, model_name, model_type='General'):
        try:
            if model_type == 'General':
                model_class = self.__retrieveModelClassByNameDynamically(model_name)
                model_class.from_pretrained(model_name, cache_dir=self.__cache_dir)
            elif model_type == 'Stable Diffusion':
                from diffusers import StableDiffusionPipeline

                StableDiffusionPipeline.from_pretrained(model_name, cache_dir=self.__cache_dir)
            return [obj for obj in self.getModels() if obj['id'] == model_name]
        except Exception as e:
            raise Exception(e)

    def is_model_exists(self, model_name):
        cache_dir_result = scan_cache_dir(cache_dir=self.__cache_dir)
        for i in cache_dir_result.repos:
            if model_name == i.repo_id:
                return True
        return False

    def removeHuggingFaceModel(self, model_name: str) -> str:
        try:
            commit_hashes = []
            cache_dir_result = scan_cache_dir(cache_dir=self.__cache_dir)
            for i in cache_dir_result.repos:
                if model_name == i.repo_id:
                    for j in i.revisions:
                        commit_hashes.append(j.commit_hash)
            delete_strategy = cache_dir_result.delete_revisions(*commit_hashes)
            print("Will free " + delete_strategy.expected_freed_size_str)
            delete_strategy.execute()
            return model_name
        except Exception as e:
            print(e)
            return ''

    def __retrieveModelClassByNameDynamically(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name, cache_dir=self.__cache_dir)
        class_name = config.architectures[0]
        # Import the module dynamically
        module = importlib.import_module('transformers')
        # Retrieve the class object from the module
        model_class = getattr(module, class_name)
        return model_class

    def getModelObject(self, model_name: str):
        return self.__retrieveModelClassByNameDynamically(model_name)


