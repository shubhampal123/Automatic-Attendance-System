from flask import Flask,render_template,request,flash,redirect,url_for
from flask_login import login_user,login_required,LoginManager,logout_user,current_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import scoped_session,sessionmaker
from sqlalchemy import create_engine
import random
import string
import csv
import datetime
#from .core import add_student_face_recognition

app=Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]='mysql+pymysql://root:root@localhost/attendance_system'
app.config["SQLALCHEMY_TRACK_MODIFICATION"]=False
app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'




from .models import User,db,Student,Attendance
db.init_app(app)

engine=create_engine('mysql+pymysql://root:root@localhost/attendance_system')
db1=scoped_session(sessionmaker(bind=engine))

login_manager=LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id)) 




@app.route('/')
def index():
    return render_template('home.html')

@app.route('/add_student',methods=["POST","GET"])
@login_required
def add_student():
    if request.method=="POST":
        rollno=request.form.get('rollno')
        email_id=request.form.get('email_id')
        name=request.form.get('name')
        branch=request.form.get('branch')
        section=request.form.get('section')
        new_student=Student(rollno=rollno,name=name,email=email_id,branch=branch,section=section)
        db.session.add(new_student)
        password=''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for i in range(8))
        new_user=User(email=email_id,password=password,name=name,is_admin=0)
        db.session.add(new_user)
        
        db.session.commit()

        add_student_face_recognition(r'facedatabase',rollno)
        
        return redirect(url_for('adminhome'))
    else:
        return render_template('add_student.html')
    

@app.route('/get_attendance')
@login_required
def get_attendance():
    email=current_user.email
    sid=Student.query.filter_by(email=email).first().id
    attendance=Attendance.query.filter_by(student_id=sid)
    tot=0
    pre=0
    for att in attendance:
        if att.status==1:
            pre+=1
        tot+=1

    percent=pre/tot
    percent*=100.0
    return render_template('get_attendance.html',attendance=attendance,total=tot,present=pre,percent=percent)

@app.route('/take_attendance')
@login_required
def take_attendance():
    take_attendance_face_recognition()
    return redirect(url_for('adminhome'))


@app.route('/update_attendance')
@login_required
def update_attendance():
    now=datetime.datetime.now()
    date_now=now.strftime("%Y-%m-%d")
    year=now.strftime("%Y")
    month=now.strftime("%m")
    day=now.strftime("%d")
    f=open('/home/shubham/Desktop/Attendance_System/Attendance/'+date_now+'.csv')
    read=csv.reader(f)
    
    for a,b in read:
        sid=Student.query.filter_by(rollno=str(a)).first().id
        att=Attendance(day=day,month=month,year=year,status=b,student_id=sid)
        db.session.add(att)

    db.session.commit()
    
    return redirect(url_for('adminhome'))
        
    
    
    
    
    

@app.route('/adminhome')
@login_required
def adminhome():
    return render_template('adminhome.html')

@app.route('/studenthome')
@login_required
def studenthome():
    return render_template('studenthome.html')

@app.route('/adminlogin',methods=["POST","GET"])
def adminlogin():
    if request.method=="POST":
        email=request.form.get('email')
        password=request.form.get('password')
        #user=db1.execute("select * from user where email=:email and password=:password and is_admin=1 ;",{"email":email,"password":password}).first()
        user=User.query.filter_by(email=email).first()
         
        if user is None:
            flash('Please check your login details and try again.')
            return redirect(url_for('adminlogin'))
        elif user.password!=password or user.is_admin==0:
            return redirect(url_for('adminlogin'))          
        else:
            login_user(user)
            
            return redirect(url_for('adminhome'))
            
    else:
        return render_template('adminlogin.html')
    

@app.route('/studentlogin',methods=["POST","GET"])
def studentlogin():
     if request.method=="POST":
        email=request.form.get('email')
        password=request.form.get('password')
        user=User.query.filter_by(email=email).first()
        if user is None:
            flash('Please check your login details and try again.')
            return redirect(url_for('studentlogin'))
        elif user.password!=password or user.is_admin==1:
            return redirect(url_for('studentlogin'))  
        else:
            login_user(user)
            return redirect(url_for('studenthome'))
            
     else:
        return render_template('studentlogin.html')

@app.route('/adminlogout')
@login_required
def adminlogout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/studentlogout')
@login_required
def studentlogout():
    logout_user()
    return redirect(url_for('index'))


def main():
    db.create_all()

if __name__=="__main__":
   print('lol')
   with app.app_context():
       main()




