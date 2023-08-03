from transformers import Trainer
import torch


class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # if self.use_amp:
            #     with autocast():
            #         loss = self.compute_loss(model, inputs)
            # else:
                loss = self.compute_loss(model, inputs)
        return (loss, None, None)
 