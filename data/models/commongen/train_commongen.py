import os

import torch
import fire

def train(model: str, device: str = "cuda", output_file: str = "commongen"):
    """Train the given `model` on the Commongen task, then store the model
    in {output_file}.pt
    
    [description]
    
    Args:
        model (str): The desired model to train. Defaults to "t5-small"
        device (str): The device to use, either "cuda" or "cpu"
        output_file (str): Path to the desired output file
    
    Returns:
        torch.device: [description]
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

        class Seq2SeqCPUTrainingArguments(Seq2SeqTrainingArguments):
            @property
            def device(self) -> torch.device:
                return torch.device("cpu")

        trainer_args = Seq2SeqCPUTrainingArguments(evaluation_strategy="no",
                                                   predict_with_generate=True,
                                                   save_strategy="epoch",
                                                   output_dir="model",
                                                   learning_rate=1e-5,
                                                   num_train_epochs=10,
                                                   fp16=False)
    else:
        trainer_args = Seq2SeqTrainingArguments(evaluation_strategy="no",
                                                predict_with_generate=True,
                                                save_strategy="epoch",
                                                output_dir="model",
                                                learning_rate=1e-5,
                                                num_train_epochs=10,
                                                fp16=True)
    trainer = Seq2SeqTrainer(model,
                             trainer_args,
                             tokenizer=tokenizer,
                             train_dataset=tr_dataset,
                             eval_dataset=vl_dataset,
                             data_collator=data_collator)
    trainer.train()

    torch.save(model.state_dict(), "{output_file}.pt")


if __name__ == '__main__':
    main()
