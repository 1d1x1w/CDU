import os.path as osp
import sys

from dassl.utils import listdir_nohidden

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class DomainNet(DatasetBase):
    """DomainNet.

        Statistics:
            - Around 580k images.
            - 345 classes.
            - 6 domains: clipart, infograph, painting, quickdraw, real, sketch.
            - URL: [
                        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
                        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
                        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
                        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
                        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
                        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
                    ].

        """

    dataset_dir = "domainnet"
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )


        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()
            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)

        return items
