def fit(self, features, labels, sample_weight=None):
  start = time.time()
  if self.clones:
    self.clfs_ = clone(self.classifiers)
    self.meta_clf_  = clone(self.meta_classifier)
  else:
    self.clfs_ = self.classifiers
    self.meta_clf_ = self.meta_classifier
    
  if self.verbose > 0:
    print('Fitting %d classifiers' % (len(self.classifiers)))
  n=1
  for clf in self.clfs_:
    if self.verbose > 1:
      print(f"Fitting classifier {n}/{len(self.clfs_)}")
      n +=1
    if sample_weight is None:
      clf.fit(features ,labels)
    else:
      clf.fit(features, labels, sample_weight)
  meta_features = self.predict_meta(features)
  if verbose >1:
    print("Fitting meta-classifier to meta_features")
  elif sparse.issparse(features):
    meta_features = sparse.hstack((features, meta_features))
  else:
    meta_features = np.hstack((features, meta_features))
  self.meta_features_ = meta_features
  if sample_weight is None:
    self.meta_clf_.fit(meta_features, labels)
  else:
    self.meta_clf_.fit(meta_features, labels, sample_weight=sample_weight)

  stop = time.time()
  if verbose > 0:
    print(f"Estimators Fit! Time Elapsed: {(stop-start)/60} minutes")

  return self

def predict_meta(self, features):
  if self.use_probability:
    probs = np.asarray([clf.predict_probs(features) for clf in self.clfs_])
    if self.average_probs:
      preds = np.average(probs, axis=0)

    else:
      preds = np.concatenate(probs, axis=1)

  else:
    preds = np.column_stack([clf.predict(features) for clf in self.clfs_])
        
  return preds

def predict_probs(self, features):
  meta_features = self.predict_meta(features)

  if self.double_down == False:
    return self.meta_clf_.predict_probs(meta_features)
        
  elif sparse.issparse(features):
    return self.meta_clf_.predict_probs(sparse.hstack((features, meta_features)))

  else:
    return self.meta_clf_.predict_probs(np.hstack((features, meta_features)))

def predict(self, features):
  meta_features = self.predict_meta(features)
  if self.double_down == False:
    return self.meta_clf_.predict(meta_features)

  elif sparse.issparse(features):
    return self.meta_clf_.predict(sparse.hstack((features, meta_features)))

  else:
    return self.meta_clf_.predict(np.hstack((features, meta_features)))

def __init__(self, classifiers=None, meta_classifier=None, 
            use_probability=False, double_down=False, 
            average_probs=False, clones=True, verbose=2):
  self.classifiers = classifiers
  self.meta_classifier = meta_classifier
  self.use_probability = use_probability
  self.double_down = double_down
  self.average_probs = average_probs
  self.clones = clones
  self.verbose = verbose
