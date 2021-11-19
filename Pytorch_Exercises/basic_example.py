import logging


class Operation:
    def __init__(self,argument_1,argument_2):
        self.argument_1 = argument_1
        self.argument_2 = argument_2
        
    def perform_operation(self):
        
        self.new_value =  self.argument_1 / self.argument_2



def main():

    logging.info("operation starting ...")

    operation_instance = Operation(5,None)
    operation_instance.perform_operation()

    logging.info("operation finished")

if __name__ == '__main__':
    main()