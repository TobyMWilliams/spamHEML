A project written in python, to securely classify emails into spam and ham classes using machine learning algorithm(SVM), 
implementing Full Homomorphic Encryption using Tenseal's library, to maintain security of email data throughout the inference pipeline. 

This method was solved using a variety of combinations of Feature type (most performant being TF-IDF),
dimensional reduction algorithm (best was a combination of chi2 and ICA, though best w.r.t time complexity was SVD).

The performance of the processing pipeline using TF-IDF features, reduced by chi2 and SVD, with inference drawn usnig SVM. 
Confusion Matrix:
            True Positives: 982
            True Negatives: 965
            False Positives: 27
            False Negatives: 26

            Other Important Metrics:
            Accuracy: 0.9735
            Sensitivity (Recall): 0.9742
            Specificity: 0.9728
            MCC: 0.9470
            
