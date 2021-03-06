from absl import app, flags, logging
import torch as th
import pytorch_lightning as pl
import os
import pathlib
import nlp
import transformers
flags.DEFINE_bool('debug',False,'')
flags.DEFINE_integer('epochs',10,'')
flags.DEFINE_integer('batch_size',8,'')
flags.DEFINE_float('lr',1e-2,'')
flags.DEFINE_float('momentum',0.9,'')
flags.DEFINE_integer('seq_length',32,'')
flags.DEFINE_string('model', 'bert-base-uncased', '')
FLAGS = flags.FLAGS


class IMBDSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)

        def _tokenize(x):
            x['input_ids'] = tokenizer.encode(x['text'],
                                              max_length=FLAGS.seq_length,
                                              pad_to_max_length=True)
            return x


        def _prepare_dataset(split):
            ds = nlp.load_dataset('imdb',split=f'{split}[:{FLAGS.batch_size if FLAGS.debug else "5%"}]')



            ds = ds.map(_tokenize)
            ds.set_format(type ='torch', columns = ['input_ids','label'])
            return ds

        self.train_ds,self.test_ds = (_prepare_dataset('train'),_prepare_dataset('test'))





    def forward(self, batch):
        pass

    def training_step(self, batch,batch_index):
        pass

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_ds,
            batch_size=FLAGS.batch_size,
            drop_last= True,
            shuffle=True
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.test_ds,
            batch_size=FLAGS.batch_size,
            drop_last= False,
            shuffle=False
        )


    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr = FLAGS.lr,
            momentum = FLAGS.momentum,
        )



def main(_):
    logging.info('main function')

    model = IMBDSentimentClassifier()
    trainer = pl.Trainer(default_root_dir = 'logs',
                         gpus = (1 if th.cuda.is_available() else 0),
                         max_epochs = FLAGS.epochs,
                         fast_dev_run=FLAGS.debug
    )

    trainer.fit(model)


if __name__ == '__main__':

    app.run(main)