"""

Custom PathManager implementation for FsDet/Detectron2.

This module defines a project-specific PathManager that avoids relying on the
global fvcore PathManager, which can cause conflicts when multiple libraries
register handlers for overlapping URL or filesystem prefixes.

Two custom handlers are registered:
    * Detectron2Handler – Resolves detectron2:// paths to the Detectron2 model zoo.
    * FsDetHandler      – Resolves fsdet:// paths to the FsDet model zoo.

These handlers support convenient referencing of remote assets using short,
namespace-based paths such as:
    detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
    fsdet://baseline/finetune/checkpoint.pth

This is a detectron2 project-specific PathManager.
We try to stay away from global PathManager in fvcore as it
introduces potential conflicts among other libraries.
"""
from iopath.common.file_io import (
    HTTPURLHandler,
    OneDrivePathHandler,
    PathHandler,
    PathManager as PathManagerBase,
)

__all__ = ["PathManager", "PathHandler"]


PathManager = PathManagerBase()


class Detectron2Handler(PathHandler):
    """
    Handler for resolving Detectron2 model zoo paths.

    Paths using the detectron2:// prefix will be automatically mapped to
    Detectron2's public file repository hosted on Facebook AI's CDN.

    Example:
        detectron2://ImageNetPretrained/MSRA/R-50.pkl
         -> https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl

    Attributes:
        PREFIX (str): URI-like namespace for model zoo references.
        S3_DETECTRON2_PREFIX (str): Base URL where Detectron2 assets are stored.
    """


    PREFIX = "detectron2://"
    S3_DETECTRON2_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    def _get_supported_prefixes(self):
        """
        Return:
            list[str]: The list of prefixes that this handler can process.
        """
        return [self.PREFIX]

    def _get_local_path(self, path):
        """
        Convert a namespace path to a real local path.

        Args:
            path (str): Path beginning with detectron2://

        Returns:
            str: Local cached path to the resolved file.
        """
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.S3_DETECTRON2_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        """
        Open a file resolved from a detectron2:// URI.

        Args:
            path (str): detectron2:// style path.
            mode (str): File access mode.
            **kwargs: Additional parameters passed to PathManager.open().

        Returns:
            file: A file-like object for reading or writing.
        """
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


class FsDetHandler(PathHandler):
    """
    Handler for resolving FsDet model zoo paths.

    Paths using the `fsdet://` prefix will be mapped to FsDet's public model
    repository.

    Example:
        fsdet://coco/base_model.pth -> http://dl.yf.io/fs-det/models/coco/base_model.pth

    Attributes:
        PREFIX (str): URI-like namespace for FsDet assets.
        URL_PREFIX (str): Base URL where FsDet models are hosted.
    """

    PREFIX = "fsdet://"
    URL_PREFIX = "http://dl.yf.io/fs-det/models/"

    def _get_supported_prefixes(self):
        """
        Return:
            list[str]: Supported URI prefixes for this handler.
        """
        return [self.PREFIX]

    def _get_local_path(self, path):
        name = path[len(self.PREFIX) :]
        return PathManager.get_local_path(self.URL_PREFIX + name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)


PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
PathManager.register_handler(Detectron2Handler())
PathManager.register_handler(FsDetHandler())
