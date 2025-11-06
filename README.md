Scripts for processing and exploring different sensor datasets related to water quality in the Brusdalsvatnet drinking water reservoir, with the intention of developing data-driven methods of forecasting water quality parameters at the offtake for the treatment plant.

Typical workflow:
1. Configure and run d_Resample to divide the full dataset into samples of a standard size. The # of samples generated will be constrained by the coverage of the available source datasets, so use the "analysis" functions to select sample dimensions which will provide a set of sufficient quantity and quality for training and testing with the input and output of interest.
2. Configure and run e_TrainTransformer to create a model for forecasting a particular set of outputs from a particular set of inputs.
3. Configure and run f_EvaluateModel to test the model's ability to predict water quality parameters, compared to several simple baseline models.