
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
db=SQLAlchemy()

class User(UserMixin,db.Model):
    __tablename__='user'
    id=db.Column(db.Integer,primary_key=True)
    email=db.Column(db.String(100),unique=True,nullable=False)
    password=db.Column(db.String(100),nullable=False)
    name=db.Column(db.String(100),nullable=False)
    is_admin=db.Column(db.Integer,nullable=False)

class Student(db.Model):
    __tablename__='student'
    id=db.Column(db.Integer,primary_key=True)
    rollno=db.Column(db.String(100),unique=True,nullable=False)
    email=db.Column(db.String(100),unique=True,nullable=False)
    name=db.Column(db.String(100),nullable=False)
    branch=db.Column(db.String(100),nullable=False)
    section=db.Column(db.String(100),nullable=False)
    attendance=db.relationship('Attendance',backref="student",cascade="all, delete-orphan",lazy='dynamic')

class Attendance(db.Model):
    __tablename__='attendance'
    id=db.Column(db.Integer,primary_key=True)
    day=db.Column(db.Integer,nullable=False)
    month=db.Column(db.Integer,nullable=False)
    year=db.Column(db.Integer,nullable=False)
    status=db.Column(db.Integer,nullable=False)
    student_id=db.Column(db.Integer,db.ForeignKey('student.id'))
    
    
    
    
    
    
