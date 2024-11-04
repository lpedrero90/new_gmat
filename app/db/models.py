from sqlalchemy import Column, String, Integer, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Key(Base):
    __tablename__ = "keys"
    id = Column(Integer, primary_key=True, autoincrement=True)
    hashed_key = Column(String, nullable=False, unique=True)
    encrypted_key = Column(String, nullable=True)

    permissions = relationship("Permission", back_populates="api_key")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    gmat_id = Column(String, unique=True, nullable=False)

    permissions = relationship("Permission", back_populates="user")
    documents = relationship("Document", back_populates="user")

class Permission(Base):
    __tablename__ = "permissions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key_id = Column(Integer, ForeignKey("keys.id"), nullable=False)
    has_pdf = Column(Boolean, default=False)
    has_sql = Column(Boolean, default=False)

    user = relationship("User", back_populates="permissions")
    api_key = relationship("Key", back_populates="permissions")


class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)

    documents = relationship("Document", back_populates="topic")


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)

    user = relationship("User", back_populates="documents")
    topic = relationship("Topic", back_populates="documents")