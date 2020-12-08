#coding=utf-8
# Copyright 2017 - 2018 Baidu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import numpy as np
import os

from abc import ABCMeta
from abc import abstractmethod

import logging
logger=logging.getLogger(__name__)



import torchvision
from torch.autograd import Variable
import torch.nn as nn

"""
The base model of the model.
"""

"""
Pytorch model
"""

class Model(object):
    """
    Base class of model to provide attack.

    Args:
        bounds(tuple): The lower and upper bound for the image pixel.
        channel_axis(int): The index of the axis that represents the color
                channel.
        preprocess(tuple): Two element tuple used to preprocess the input.
            First substract the first element, then divide the second element.
    """
    __metaclass__ = ABCMeta

    def __init__(self, bounds, channel_axis, preprocess=None):
        assert len(bounds) == 2
        assert channel_axis in [0, 1, 2, 3]

        self._bounds = bounds
        self._channel_axis = channel_axis

        # Make self._preprocess to be (0,1) if possible, so that don't need
        # to do substract or divide.
        if preprocess is not None:
            sub, div = np.array(preprocess)
            if not np.any(sub):
                sub = 0
            if np.all(div == 1):
                div = 1
            assert (div is None) or np.all(div)
            self._preprocess = (sub, div)
        else:
            self._preprocess = (0, 1)

    def bounds(self):
        """
        Return the upper and lower bounds of the model.
        """
        return self._bounds

    def channel_axis(self):
        """
        Return the channel axis of the model.
        """
        return self._channel_axis

    def _process_input(self, input_):
        res = None
        sub, div = self._preprocess
        if np.any(sub != 0):
            res = input_ - sub
        if not np.all(sub == 1):
            if res is None:  # "res = input_ - sub" is not executed!
                res = input_ / div
            else:
                res /= div
        if res is None:  # "res = (input_ - sub)/ div" is not executed!
            return input_
        return res

    @abstractmethod
    def predict(self, data):
        """
        Calculate the prediction of the data.

        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).

        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        """
        Determine the number of the classes

        Return:
            int: the number of the classes
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.

        Args:
            data(numpy.ndarray): input data with shape (size, height, width,
            channels).
            label(int): Label used to calculate the gradient.

        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        raise NotImplementedError

#直接加载pb文件
class PytorchModel(Model):


    def __init__(self,
                 model,
                 loss,
                 bounds,
                 channel_axis=3,
                 nb_classes=10,
                 preprocess=None,
                 device=None):

        import torch


        if preprocess is None:
            preprocess = (0, 1)

        super(PytorchModel, self).__init__(
            bounds=bounds, channel_axis=channel_axis, preprocess=preprocess)


        self._model = model

        #暂时不支持自定义loss
        self._loss=loss

        self._nb_classes=nb_classes
        if not device:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == -1:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cuda:{}".format(device))

        print(self._device)

        logger.info("Finish PytorchModel init")

    #返回值为标量
    def predict(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """

        import torch

        scaled_data = self._process_input(data)

        scaled_data = torch.from_numpy(scaled_data).to(self._device)


        # Run prediction
        predict = self._model(scaled_data)
        #if A3C choose action output, don't care about value for evaluation
        if type(predict)==tuple:
            predict = predict[1]
        predict = np.squeeze(predict, axis=0)

        predict=predict.detach()

        predict=predict.cpu().numpy().copy()

        #logging.info(predict)

        return predict

    #返回值为tensor
    def predict_tensor(self, data):
        """
        Calculate the prediction of the data.
        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).
        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """

        import torch

        scaled_data = self._process_input(data).to(self._device)

        #scaled_data = torch.from_numpy(scaled_data)


        # Run prediction
        predict = self._model(scaled_data)
        #predict = np.squeeze(predict, axis=0)

        #predict=predict.detach()

        #predict=predict.numpy()

        #logging.info(predict)

        return predict

    def num_classes(self):
        """
            Calculate the number of classes of the output label.
        Return:
            int: the number of classes
        """

        return self._nb_classes

    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.
        Args:
            data(numpy.ndarray): input data with shape (size, height, width,
            channels).
            label(int): Label used to calculate the gradient.
        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """

        import torch

        scaled_data = self._process_input(data)

        #logging.info(scaled_data)

        scaled_data = torch.from_numpy(scaled_data).to(self._device)
        scaled_data.requires_grad = True

        label = np.array([label])
        label = torch.from_numpy(label).to(self._device)
        #deal with multiple outputs
        try:
            output=self.predict_tensor(scaled_data).to(self._device)
        except(AttributeError):
            output = self.predict_tensor(scaled_data)[1].to(self._device)
        #loss=self._loss(output, label)
        ce = nn.CrossEntropyLoss()
        loss=-ce(output, label)

        #计算梯度
        # Zero all existing gradients
        self._model.zero_grad()
        loss.backward()

        #技巧 梯度也是tensor 需要转换成np
        grad = scaled_data.grad.cpu().numpy().copy()


        return grad.reshape(scaled_data.shape)

    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        return self._predict_program.block(0).var(self._predict_name).op.type
