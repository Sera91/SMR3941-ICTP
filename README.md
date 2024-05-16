# SMR3941-ICTP
Official repo for the ICTP/School on Applied ML

To connect to HPC systems at CINECA you can use the command:

ssh -o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "LogLevel ERROR" $USER@login.leonardo.cineca.it

where $USER is your account on CINECA

The horrible string '-o "StrictHostKeyChecking=no" -o "UserKnownHostsFile=/dev/null" -o "LogLevel ERROR"'  is needed to avoid the annoying error message about know_host file
