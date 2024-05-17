from rest_framework import serializers
from .models import User, Article

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'heading', 'keywords', 'published_date', 'text']

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'age', 'gender', 'interest')
        extra_kwargs = {
            'password': {'write_only': True},
            'email': {'required': True}
        }

    def create(self, validated_data):
        # Kullanıcı oluşturma
        user = User(
            username=validated_data['username'],
            email=validated_data['email'],
            age=validated_data.get('age'),
            gender=validated_data.get('gender'),
            interest=validated_data.get('interest'),
        )
        user.set_password(validated_data['password'])  # Şifreyi güvenli bir şekilde hash'ler
        user.save()
        return user

    def update(self, instance, validated_data):
        # Kullanıcı bilgilerini güncelleme
        instance.username = validated_data.get('username', instance.username)
        instance.email = validated_data.get('email', instance.email)
        instance.age = validated_data.get('age', instance.age)
        instance.gender = validated_data.get('gender', instance.gender)
        instance.interest = validated_data.get('interest', instance.interest)

        password = validated_data.get('password')
        if password:
            instance.set_password(password)  # Şifreyi güvenli bir şekilde hash'ler
        instance.save()
        return instance
