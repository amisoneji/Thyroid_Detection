import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pickle

class prediction:

    def __init__(self,path):
        self.file_object=open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer=logger.App_logger()
        self.pred_data_val=Prediction_Data_validation(path)


    def predict_from_model(self):
        try:
            self.pred_data_val.deletePredictionFile()
            self.log_writer.log(self.file_object,'start of prediction')
            data_getter=data_loader_prediction.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data=preprocessor.drop_unnecessary_columns(data,['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured',
                                                        'FTI_measured', 'TBG_measured', 'TBG', 'TSH'])

            data=preprocessor.replaceInvalidValueWithNULL(data)

            data=preprocessor.EncodeCategoricalValuePrediction(data)

            is_null_present=preprocessor.is_null_present(data)
            if (is_null_present):
                data=preprocessor.impute_missing_values(data)

            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model("KMeans")

            clusters=kmeans.predict(data)
            data["clusters"]=clusters
            clusters=data["clusters"].unique()

            result = []  # initialize balnk list for storing predicitons
            with open('EncoderPickle/enc.pickle', 'rb') as file:  # let's load the encoder pickle file to decode the values
                encoder = pickle.load(file)

            for i in clusters:
                cluster_data=data[data["clusters"]==i]
                cluster_data=cluster_data.drop(["clusters"],axis=1)
                model_name=file_loader.find_correct_model_file(i)
                model=file_loader.load_model(model_name)
                for val in (encoder.inverse_transform(model.predict(cluster_data))):
                    result.append(val)
            result = pandas.DataFrame(result, columns=['Predictions'])
            path = "Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv", header=True)  # appends result to prediction file
            self.log_writer.log(self.file_object, 'End of Prediction')

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex


        return path


