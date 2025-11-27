import importlib
from os import path as osp
from basicsr.utils import scandir, get_root_logger

# automatically scan and import model modules for registry
# scan all the files that end with '_model.py' under the model folder
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'basicsr.models.{file_name}') for file_name in model_filenames]

def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model