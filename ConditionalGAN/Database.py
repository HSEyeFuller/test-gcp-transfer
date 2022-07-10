#!/usr/bin/python
# -*- coding: utf-8 -*-

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

from firebase_admin import credentials, initialize_app, storage, firestore


class Database:
    
    def __init__(self):
        self.db, self.bucket = self.fetchInstance()
        
        
    def fetchInstance(self):
        try:
            cred = credentials.Certificate('hseye-fff80-firebase-adminsdk-fm7tr-7eb812f1b5.json')
            initialize_app(cred, {
                'storageBucket': 'hseye-fff80.appspot.com'
            })
            firebase_admin.initialize_app(cred)
        except ValueError:
            print ("Instance already initialized")

        return firestore.client(), storage.bucket()
    
    
    def uploadSession(self, session):
        print("Adding Session...")
        name = session["name"]
        doc_ref = self.db.collection(u'CGAN').document(name)
        doc_ref.set(session)
        print("Session Added")
        
        
    def fetchDate(self):
        return firestore.SERVER_TIMESTAMP
    
    def fetchModel(self, model):
        doc_ref = self.db.collection(u'CGAN').document(model)

        doc = doc_ref.get()
        if doc.exists:
            print(doc)
            return doc.to_dict()
        else:
            return "NO SUCH DOCUMENT"
        
    
    def setOptimumCheckpoint(self, checkpoint, label):
        db_ref = self.db.collection(u'CGAN').document(label)

        db_ref.set({
            'optimalCheckpoint': checkpoint
        }, merge=True)
        
        print("Updated Optimal Checkpoint")

    
    def uploadModel(self, jobName):
          print("TEST")
#         print(jobName + "/model.h5")
          blob = self.bucket.blob(jobName)
#         blob.upload_from_filename(fileName)
#         print("Uploaded Model")

          blob.upload_from_filename(jobName)
          print("Uploaded Model")