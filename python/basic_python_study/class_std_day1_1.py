class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

user = Person("John", 36)

print(user.name)
print(user.age)

#--------------------------------------

class Person2:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def __str__(self):
    return f"{self.name}({self.age})"

user2 = Person2("John", 36)

print(user2)







