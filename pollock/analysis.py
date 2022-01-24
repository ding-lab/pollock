

## def generate_report_for_dataset(self, ds, y):
##     """Generate classification report for dataset"""
##     probs = self.predict(ds)
##     predictions = np.argmax(probs, axis=1).flatten()
##     predicted_labels = [self.class_names[i] for i in predictions]
##     groundtruth = np.asarray([int(i) for i in y])
##     groundtruth_labels = [self.class_names[i] for i in groundtruth]
## 
##     report = classification_report(groundtruth,
##             predictions, target_names=self.class_names,
##             labels=list(range(len(self.class_names))), output_dict=True)
## 
##     # if we need to
##     c_df = pollock_analysis.get_confusion_matrix(predictions,
##             groundtruth, self.class_names, show=False)
## 
##     d = {
##         'metrics': report,
##         'probabilities': probs,
##         'prediction_labels': predicted_labels,
##         'groundtruth_labels': groundtruth_labels,
##         'confusion_matrix': c_df.values,
##         }
##     return
