"""Lệnh quản trị: python manage.py create-admin [--username admin] [--password ...]"""
import argparse
import getpass
import os
import sys

from sqlalchemy import select

from app import config
from app.db import Base, SessionLocal, engine
from app.models import User
from app.security import hash_password


def create_admin(username, password):
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    if len(password) < 8:
        sys.exit("Mật khẩu cần tối thiểu 8 ký tự.")
    with SessionLocal() as db:
        u = db.scalar(select(User).where(User.username == username))
        if u:
            sys.exit(f"Tài khoản '{username}' đã tồn tại. Đổi mật khẩu ở trang Quản trị hoặc chọn --username khác.")
        db.add(User(username=username, password_hash=hash_password(password), role="admin"))
        db.commit()
    print(f"Đã tạo admin '{username}'.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="manage.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-admin", help="tạo tài khoản admin")
    c.add_argument("--username", default="admin")
    c.add_argument("--password", help="bỏ trống: lấy từ biến ADMIN_PASSWORD hoặc hỏi")
    args = ap.parse_args(argv)
    pw = args.password or os.getenv("ADMIN_PASSWORD")
    if not pw:
        pw = getpass.getpass("Mật khẩu: ")
        if pw != getpass.getpass("Nhập lại: "):
            sys.exit("Hai lần nhập không khớp.")
    create_admin(args.username, pw)


if __name__ == "__main__":
    main()
