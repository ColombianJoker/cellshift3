
import pandas
from cellshift import CS

df = pandas.DataFrame({
  "id":[10*(1000-i) for i in range(12)],
  "names":[ "Armando", "Benito", "Carlos", "Daniel",
            "Ernesto", "Felipe", "Gustavo", "Hernando",
            "Ignacio", "Juan", "Karl", "Luis", ],
  "mails":[ "armando@gmail.com", "benito@hotmail.com", "carlos@yahoo.com", "daniel@gmx.net",
            "ernesto@gmx.net", "felipe@yahoo.com", "gustavo@hotmail.com", "hernando@gmai.com",
            "ignacio@gmail.com", "juan@hotmail.com", "karl@yahoo.com", "luis@gmx.net",],
})
ob = CS(df)
ob.data.show()
# ob.remove_rows("names", condition="?=='Armando'")
ob.add_masked_column("id", "masked_1", mask_right=1).data.show()
ob.add_masked_column("masked_1", "masked_2", mask_left=1, mask_char='?').data.show()
ob.add_masked_column("names", mask_left=2, mask_char='⌘').data.show()
ob.add_masked_mail_column("mails", mask_domain=True, 
  domain_choices=["a.org", "b.com", "c.net", "d.co"],).data.show()
