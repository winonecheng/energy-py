import tensorflow as tf


# action = charging is positive, discharging is negative [MW]
actions = tf.constant([0.5, 4.0, -0.5])
actions = tf.reshape(actions, (3, 1))

old_charge = tf.constant([0.0, 1.0, 1.0])
old_charge = tf.reshape(old_charge, (3, 1))

capacity = 4.0
eff = 0.9

new_charge = tf.clip_by_value(
    tf.add(old_charge, actions), tf.zeros_like(old_charge), tf.fill(actions.shape, capacity)
)

#  hourly basis
gross_power = new_charge - old_charge

discharging = tf.where(gross_power < 0, gross_power, tf.zeros_like(gross_power))

loss = tf.abs(discharging * (1 - eff))

charge = old_charge + gross_power

