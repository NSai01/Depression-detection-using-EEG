from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.hashers import make_password, check_password
import os, io, base64, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .models import (
    DatasetRecord,
    AlgorithmResult,
    PredictionHistory,
    UserProfile
)

BASE_DIR = settings.BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR,'model')
_ml_state={}


def get_plot_base64(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight')
    buf.seek(0)
    image=base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return image


def login_view(request):

    if request.session.get('logged_in'):
        return redirect('dashboard')

    if request.method=='POST':
        username=request.POST.get('username','').strip()
        password=request.POST.get('password','').strip()

        try:
            user=UserProfile.objects.get(username=username)

            if check_password(password,user.password):
                request.session['logged_in']=True
                request.session['username']=username
                return redirect('dashboard')

            return render(request,'depression_detection/login.html',{
                'error':'Invalid username or password'
            })

        except UserProfile.DoesNotExist:
            return render(request,'depression_detection/login.html',{
                'error':'Invalid username or password'
            })

    return render(request,'depression_detection/login.html')


def register_view(request):

    if request.method=='POST':
        username=request.POST.get('username').strip()
        email=request.POST.get('email').strip()
        password=request.POST.get('password').strip()
        confirm=request.POST.get('confirm_password').strip()

        if password!=confirm:
            return render(request,'depression_detection/register.html',{
                'error':'Passwords do not match'
            })

        if UserProfile.objects.filter(username=username).exists():
            return render(request,'depression_detection/register.html',{
                'error':'Username already exists'
            })

        UserProfile.objects.create(
            username=username,
            email=email,
            password=make_password(password)
        )

        return render(request,'depression_detection/register.html',{
            'success':'Account created successfully'
        })

    return render(request,'depression_detection/register.html')


def logout_view(request):
    request.session.flush()
    _ml_state.clear()
    return redirect('login')


def dashboard(request):
    if not request.session.get('logged_in'):
        return redirect('login')

    return render(request,'depression_detection/dashboard.html')


def upload_dataset(request):

    if not request.session.get('logged_in'):
        return redirect('login')

    context = {}

    if request.method == 'POST' and request.FILES.get('dataset_file'):

        f = request.FILES['dataset_file']

        save_path = os.path.join(
            BASE_DIR,
            'media',
            'uploads',
            f.name
        )

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        with open(save_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        try:

            dataset = pd.read_csv(save_path)

            dataset['Label'] = dataset['Label'].replace(
                2.0,
                1.0
            )

            counts = dataset.groupby(
                'Label'
            ).size()

            normal = int(
                counts.get(0.0,0)
            )

            depressed = int(
                counts.get(1.0,0)
            )

            total = len(dataset)

            request.session['dataset_path'] = save_path
            request.session['dataset_loaded'] = True

            DatasetRecord.objects.create(
                filename=f.name,
                total_records=total,
                normal_count=normal,
                depressed_count=depressed
            )

            fig, ax = plt.subplots(
                figsize=(6,4)
            )

            ax.bar(
                ['Normal','Depressed'],
                [normal,depressed]
            )

            ax.set_title(
                'Dataset Distribution'
            )

            ax.set_ylabel(
                'Count'
            )

            chart = get_plot_base64(fig)

            context = {
                'success': True,

                'filename': f.name,

                'total': total,

                'normal': normal,

                'depressed': depressed,

                'chart': chart
            }

        except Exception as e:

            context['error'] = (
                f'Error reading dataset: {str(e)}'
            )

    return render(
        request,
        'depression_detection/upload_dataset.html',
        context
    )


def features_extraction(request):

    if not request.session.get('logged_in'):
        return redirect('login')

    context = {}

    if request.method == 'POST':

        path = request.session.get('dataset_path')

        if not path:
            context['error'] = 'Please upload dataset first'
            return render(
                request,
                'depression_detection/features_extraction.html',
                context
            )

        try:
            dataset = pd.read_csv(path)

            dataset.fillna(0, inplace=True)
            dataset['Label'] = dataset['Label'].replace(2.0,1.0)

            data = dataset.values

            X = data[:, :-1]
            Y = data[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                Y,
                test_size=0.2,
                random_state=42,
                stratify=Y
            )

            _ml_state['X'] = X
            _ml_state['Y'] = Y
            _ml_state['X_train'] = X_train
            _ml_state['X_test'] = X_test
            _ml_state['y_train'] = y_train
            _ml_state['y_test'] = y_test

            request.session['features_extracted']=True

            context = {
                'success': True,

                'total_features': X.shape[1],

                'total_records': X.shape[0],

                'train_records': X_train.shape[0],

                'test_records': X_test.shape[0],

                'sample_features':
                str(X[:3]).replace('\n','<br>')
            }

        except Exception as e:

            context['error'] = str(e)

    return render(
        request,
        'depression_detection/features_extraction.html',
        context
    )


def calculate_metrics(name,predict,test_y):

    p=precision_score(test_y,predict,average='macro',zero_division=0)*100
    r=recall_score(test_y,predict,average='macro',zero_division=0)*100
    f=f1_score(test_y,predict,average='macro',zero_division=0)*100
    a=accuracy_score(test_y,predict)*100

    cm=confusion_matrix(test_y,predict)

    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt='g',ax=ax)
    chart=get_plot_base64(fig)

    return {
        'accuracy':round(a,2),
        'precision':round(p,2),
        'recall':round(r,2),
        'fscore':round(f,2),
        'confusion_chart':chart
    }


def run_svm(request):

    context={}

    if request.method=='POST':

        if _ml_state.get('X_train') is None:
            context['error']='Run feature extraction first'
            return render(request,'depression_detection/run_svm.html',context)

        svm=SVC(
            kernel='linear',
            class_weight='balanced'
        )

        svm.fit(_ml_state['X_train'],_ml_state['y_train'])

        pred=svm.predict(_ml_state['X_test'])

        metrics=calculate_metrics(
            'SVM',
            pred,
            _ml_state['y_test']
        )

        AlgorithmResult.objects.create(
            algorithm='SVM',
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            fscore=metrics['fscore']
        )

        _ml_state['svm_metrics']=metrics

        context={
            'success':True,
            'metrics':metrics
        }

    return render(request,'depression_detection/run_svm.html',context)


def run_cnn(request):

    if not request.session.get('logged_in'):
        return redirect('login')

    context = {}

    if request.method == 'POST':

        if 'X' not in _ml_state:
            context['error'] = (
                'Please run Features Extraction first.'
            )

            return render(
                request,
                'depression_detection/run_cnn.html',
                context
            )

        try:

            X = _ml_state['X'].copy()
            Y = _ml_state['Y'].copy()

            X = X[:,0:972]

            # normalize data
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            XX = X.reshape(
                X.shape[0],
                18,
                18,
                3
            )

            YY = to_categorical(Y)

            X_train1, X_test1, y_train1, y_test1 = train_test_split(
                XX,
                YY,
                test_size=0.2,
                random_state=42,
                stratify=Y
            )

            classes=np.unique(Y)

            weights=compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=Y
            )

            class_weights=dict(
                enumerate(weights)
            )

            os.makedirs(
                MODEL_DIR,
                exist_ok=True
            )

            model_json_path = os.path.join(
                MODEL_DIR,
                'model.json'
            )

            model_weights_path = os.path.join(
                MODEL_DIR,
                'model_weights.h5'
            )

            # delete old files manually once before testing
            if (
                os.path.exists(model_json_path)
                and
                os.path.exists(model_weights_path)
            ):

                with open(model_json_path,'r') as json_file:
                    cnn=model_from_json(json_file.read())

                cnn.load_weights(model_weights_path)

                cnn.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

            else:

                cnn=Sequential()

                cnn.add(
                    Conv2D(
                        32,
                        (3,3),
                        activation='relu',
                        padding='same',
                        input_shape=(18,18,3)
                    )
                )

                cnn.add(MaxPooling2D((2,2)))

                cnn.add(
                    Conv2D(
                        64,
                        (3,3),
                        activation='relu',
                        padding='same'
                    )
                )

                cnn.add(MaxPooling2D((2,2)))

                cnn.add(
                    Conv2D(
                        128,
                        (3,3),
                        activation='relu',
                        padding='same'
                    )
                )

                cnn.add(MaxPooling2D((2,2)))

                cnn.add(Flatten())

                cnn.add(
                    Dense(
                        256,
                        activation='relu'
                    )
                )

                cnn.add(Dropout(0.5))

                cnn.add(
                    Dense(
                        128,
                        activation='relu'
                    )
                )

                cnn.add(
                    Dense(
                        y_test1.shape[1],
                        activation='softmax'
                    )
                )

                cnn.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                early=EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )

                hist=cnn.fit(
                    X_train1,
                    y_train1,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test1,y_test1),
                    callbacks=[early],
                    class_weight=class_weights,
                    verbose=1
                )

                cnn.save_weights(model_weights_path)

                with open(model_json_path,'w') as file:
                    file.write(cnn.to_json())

                with open(
                    os.path.join(MODEL_DIR,'history.pckl'),
                    'wb'
                ) as f:
                    pickle.dump(hist.history,f)

            _ml_state['cnn']=cnn

            predict=cnn.predict(X_test1)
            predict=np.argmax(predict,axis=1)
            test_y=np.argmax(y_test1,axis=1)

            metrics=calculate_metrics(
                'CNN',
                predict,
                test_y
            )

            print(metrics)

            AlgorithmResult.objects.create(
                algorithm='CNN',
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                fscore=metrics['fscore']
            )

            _ml_state['cnn_metrics']=metrics

            context={
                'success':True,
                'metrics':metrics,
                'algorithm':'CNN'
            }

        except Exception as e:
            context['error']=f'Error running CNN: {str(e)}'

    return render(
        request,
        'depression_detection/run_cnn.html',
        context
    )



def predict_depression(request):
    if not request.session.get('logged_in'):
        return redirect('login')

    context = {}
    if request.method == 'POST' and request.FILES.get('test_file'):
        if 'cnn' not in _ml_state:
            context['error'] = 'Please run CNN Algorithm first to load the model.'
            return render(request, 'depression_detection/predict.html', context)

        f = request.FILES['test_file']
        save_path = os.path.join(BASE_DIR, 'media', 'uploads', f.name)
        with open(save_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        try:
            dataset = pd.read_csv(save_path)
            dataset = dataset.values
            testData = dataset[:, 0:972]
            test_X = testData.reshape(testData.shape[0], 18, 18, 3)

            cnn = _ml_state['cnn']
            predict = cnn.predict(test_X)
            predict = np.argmax(predict, axis=1)

            labels = ['Normal', 'Depressed']
            results = []
            for i in range(len(predict)):
                result = labels[predict[i]]
                results.append({'index': i + 1, 'result': result})
                PredictionHistory.objects.create(
                    signal_filename=f.name,
                    prediction_result=result
                )

            context = {
                'success': True,
                'filename': f.name,
                'results': results,
                'total': len(results),
                'normal_count': sum(1 for r in results if r['result'] == 'Normal'),
                'depressed_count': sum(1 for r in results if r['result'] == 'Depressed'),
            }
        except Exception as e:
            context['error'] = f'Error during prediction: {str(e)}'

    return render(request, 'depression_detection/predict.html', context)


def comparison_graph(request):
    if not request.session.get('logged_in'):
        return redirect('login')

    context = {}
    svm_metrics = _ml_state.get('svm_metrics')
    cnn_metrics = _ml_state.get('cnn_metrics')

    # Also try from DB
    svm_db = AlgorithmResult.objects.filter(algorithm='SVM').last()
    cnn_db = AlgorithmResult.objects.filter(algorithm='CNN').last()

    if svm_db and cnn_db:
        svm_m = {'accuracy': svm_db.accuracy, 'precision': svm_db.precision,
                 'recall': svm_db.recall, 'fscore': svm_db.fscore}
        cnn_m = {'accuracy': cnn_db.accuracy, 'precision': cnn_db.precision,
                 'recall': cnn_db.recall, 'fscore': cnn_db.fscore}

        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        svm_vals = [svm_m['accuracy'], svm_m['precision'], svm_m['recall'], svm_m['fscore']]
        cnn_vals = [cnn_m['accuracy'], cnn_m['precision'], cnn_m['recall'], cnn_m['fscore']]

        x = np.arange(len(categories))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, svm_vals, width, label='SVM', color='#2196F3')
        bars2 = ax.bar(x + width/2, cnn_vals, width, label='CNN', color='#4CAF50')
        ax.set_ylabel('Score (%)')
        ax.set_title('SVM vs CNN Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 115)
        for bar in bars1:
            ax.annotate(f'{bar.get_height():.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        for bar in bars2:
            ax.annotate(f'{bar.get_height():.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
        chart = get_plot_base64(fig)
        context = {
            'success': True,
            'chart': chart,
            'svm': svm_m,
            'cnn': cnn_m,
        }
    else:
        context['error'] = 'Please run both SVM and CNN algorithms first.'

    return render(request, 'depression_detection/comparison_graph.html', context)


def prediction_history(request):
    if not request.session.get('logged_in'):
        return redirect('login')
    records = PredictionHistory.objects.all().order_by('-predicted_at')
    return render(request, 'depression_detection/prediction_history.html', {'records': records})
