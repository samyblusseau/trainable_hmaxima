
----------------------

TRP1 experiments

----------------------


- Download the TRP1 dataset from https://bit.ly/melanocytesTRP1 to the 'trp1' directory, and unzip it.

- Clone the Morpholoayers project just outside the trainable_hmaxima directory: git clone https://github.com/Jacobiano/morpholayers.git

- Run the preprocess_trp1.sh script.
Note that if you only wish to runcounting-loss-only experiments, you may comment the lines 109 and below in preprocess_trp1.py, since that part is meant to compute optimal h values anf the corresponding h-reconstruction, not necessary for the counting loss.


For *** training *** :

- Edit train.sh: make sure data_type=trp1, and set the other options as desired (number of epochs, modified backpropagation mode, experiment number, N value, counting loss only or not, alpha and beta values)

- Run the train.sh script.


