from paddle3d.utils.logger import logger


class QAT(object):
    def __init__(self, quant_config, print_model=False):
        self.quant_config = quant_config
        self.print_model = print_model

    def __call__(self, model):
        try:
            import paddleslim
        except:
            raise ImportError("paddleslim module not found")

        if self.print_model:
            logger.info("model before quant")
            logger.info(model)

        self.quanter = paddleslim.QAT(config=self.quant_config)
        self.quanter.quantize(model)

        if self.print_model:
            logger.info("model after quant")
            logger.info(model)

        return model

    def save_quantized_model(self, model, path, input_spec, **kwargs):
        self.quanter.save_quantized_model(
            model=model, path=path, input_spec=input_spec, **kwargs)
