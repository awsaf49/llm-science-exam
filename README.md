# Kaggle - LLM Science Exam
> Use LLMs to answer difficult science questions

<img src="https://www.kaggle.com/competitions/54662/images/header">


## Background

There are already some excellent notebooks demonstrating the use of HuggingFace's `AutoModelForMultipleChoice` for MultipleChoice tasks in [Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam) competition. However, it is challenging to comprehend the underlying mechanisms inside the model. This led me to create this notebook, which is centered around building a **MultipleChoice** model from the ground up, using the standard classifier from **KerasNLP**. In this notebook, I also use the multi-backend **KerasCore** alongside **KerasNLP**.

> Furthermore, as time progresses, it's likely that larger datasets will become available, in which **TPUs** will be invaluable for training large models on these large datasets.

## Kaggle Notebooks

* **training**: [LLM Science Exam: KerasCore + KerasNLP [TPU]](https://www.kaggle.com/code/awsaf49/llm-science-exam-kerascore-kerasnlp-tpu)
* **inference**: [LLM Science Exam: KerasCore + Keras [Infer]](https://www.kaggle.com/awsaf49/llm-science-exam-kerascore-keras-infer)

> **Note**: Train and Inference notebooks are also available in the [`notebooks`](/notebooks/) folder.

## Model Architecture

In the image below, you'll find `token_ids` on the left and corresponding `padding_masks` on the right:
![Model Architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3574256%2Fd9371a8b841160b85fee579cd8da3a25%2Fmodel_arch.png?generation=1691927852425630&alt=media)

## Augmentation

I also tried a fun augmentation, `ShuffleOptions`. This approach involves shuffling the answer options of each question. For instance, options `[A, B, C]` would be transformed into `[C, A, B]`. The purpose behind this augmentation is to ensure that the model doesn't focus on the positions of the options.

## Tracking with WandB

You can track the all experiments [here](https://wandb.ai/awsaf49/llm-science-exam-public)

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3574256%2F8d55d2c211825b4b496b85d05613ec29%2Fwb-llm.JPG?generation=1691929003378546&alt=media">

## Known Issues

* Setting backend `tensorflow` leads to OOM in RAM which is very weird. You can solve it by either using `jax` backend or using `tf.keras` instead of `keras`.
* Currently `TPU` is throwing an error with `tensorflow`. You can use `jax` backend with `keras_core` to resolve this issue.