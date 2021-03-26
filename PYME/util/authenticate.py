"""
Authentication for PYME webapps.
--------------------------------

**This is a work in progress.**

Our authentication scheme / security is based around 3 principles:

1) prevent casual misuse (in general, the consequences of a hack are comparatively low)
2) maintain good performance
3) protect user credentials in the case of a breach (accepting that many users re-use passwords)

This has lead to the following design choices:

- use jwt tokens for authentication
- store hashed passwords, rather than clear text

We have one remaining issue, namely that the majority of our connections are still http rather than https (for both
convenience and, in the case of the cluster, performance). As a consequence our passwords travel in clear text on
the network, and are vulnerable to a man-in-the-middle attack. With the current use cases (connections between nodes on
the same switch, behind an institutional firewall) the risk is low, but it would still be strongly advisable to **use a
unique, "burner"** password.
 
If we deploy more widely, we should look at using https for the UI bits and deploying tokens for the cluster in a secure
manner (e.g. SSH).

"""
import PYME.config
import sqlite3
import os
import binascii
import hashlib
import datetime
import jwt

from six.moves import input

auth_db_path = os.path.join(PYME.config.user_config_dir, '.auth')

def _init_db():
    auth_db = sqlite3.connect(auth_db_path)
    with auth_db as con:
        tableNames = [a[0] for a in con.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall()]
        
        if not 'users' in tableNames:
            con.execute("CREATE TABLE users (email TEXT NOT NULL UNIQUE , password TEXT NOT NULL)")
            
        #create secrets (salts) for this computer if not present
        if not 'secrets' in tableNames:
            con.execute("CREATE TABLE secrets (type TEXT NOT NULL UNIQUE , secret TEXT NOT NULL )")
            con.execute("INSERT INTO secrets VALUES ('SALT', ?)", (binascii.hexlify(os.urandom(24)),))
    
    auth_db.close()
            
_init_db()

def get_salt():
    auth_db = sqlite3.connect(auth_db_path)
    salt = auth_db.execute("SELECT secret FROM secrets WHERE type='SALT'").fetchone()[0]
    auth_db.close()
    
    return salt

def _get_token_secret():
    try:
        auth_db = sqlite3.connect(auth_db_path)
        secret = str(auth_db.execute("SELECT secret FROM secrets WHERE type='TOKEN'").fetchone()[0])
        auth_db.close()
        return secret
    except TypeError:
        # generate a secret the first time we want one
        secret = binascii.hexlify(os.urandom(24))
        with auth_db as con:
            con.execute("INSERT INTO secrets VALUES ('TOKEN', ?)", (secret,))
        
        return str(secret)
        
    
def hash_password(password):
    return binascii.hexlify(hashlib.pbkdf2_hmac('sha256', password.encode('utf8'), get_salt().encode('utf8'), 100000))

def authenticate_hash(email, password_hash):
    auth_db = sqlite3.connect(auth_db_path)
    hash = auth_db.execute("SELECT password FROM users WHERE email=?", (email,)).fetchone()[0]
    auth_db.close()
    return password_hash==hash

def authenticate(email, password):
    return authenticate_hash(email, hash_password(password))

def add_user(email, password):
    if len(password) < 8:
        raise ValueError('Minimum password length is 8 characters')

    auth_db = sqlite3.connect(auth_db_path)
    with auth_db:
        auth_db.execute("INSERT INTO users VALUES(?,?)", (email, hash_password(password)))
    auth_db.close()
        
def _change_password(email, password):
    """ DO NOT EXPOSE IN UI """
    if len(password) < 8:
        raise ValueError('Minimum password length is 8 characters')

    auth_db = sqlite3.connect(auth_db_path)
    with auth_db:
        auth_db.execute("UPDATE users SET password=? WHERE email=?", (hash_password(password), email))
    auth_db.close()

def _delete_user(email):
    """ DO NOT EXPOSE IN UI """
    auth_db = sqlite3.connect(auth_db_path)
    with auth_db:
        auth_db.execute("DELETE FROM users WHERE email=?", (email,))
    auth_db.close()
        
def get_token(email, password, lifetime=datetime.timedelta(days=1), **kwargs):
    if authenticate(email, password):
        payload = dict(kwargs)
        payload['email'] = email
        if not lifetime is None:
            payload['exp'] = datetime.datetime.utcnow() + lifetime
            
        return jwt.encode(payload, _get_token_secret())
    else:
        raise RuntimeError('Invalid password')
    
def validate_token(token):
    return jwt.decode(token, _get_token_secret())
        
        
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PYME password management")
    parser.add_argument('-a', '--add', help='add a user', action='store_true', default=False)
    parser.add_argument('-p', '--password', help='change password', action='store_true', default=False)
    parser.add_argument('email', type=str)
    
    args = parser.parse_args()
    
    if args.add:
        print('Enter password for new user [%s]:' % args.email)
        pw = input()
        add_user(args.email, pw)
        
        exit(0)
        
    if args.password:
        print('Change password for user [%s]:' % args.email)
        print('old password:')
        pw = input()
        if authenticate(args.email, pw):
            print('new password:')
            pw = input()
            _change_password(args.email, pw)
            print('password changed')
            exit(0)
        else:
            print('password incorrect')
            exit(1)
            
            
if __name__ == '__main__':
    main()